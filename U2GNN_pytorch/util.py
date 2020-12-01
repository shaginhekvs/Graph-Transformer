import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
import sklearn
import itertools
from dgl.data import TUDataset
from sklearn.neighbors import kneighbors_graph
from collections import Counter

from sklearn.model_selection import StratifiedKFold

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""

def get_edges_mat(g):
    edges = [list(pair) for pair in g.edges()]
    edges.extend([[i, j] for j, i in edges])
    #print(edges)
    return np.transpose(np.array(edges, dtype=np.int32), (1,0))

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0
        self.knn_g = None
        self.attr_gs = []
        self.edge_mat_knn = None
        self.edge_mat_attrs = []

def build_kneighbors(X, knn=True, n_neighbors=20):
    if knn:
        A = kneighbors_graph(X, n_neighbors, include_self=True)
        A = np.array(A.todense())
        A = (A + A.T)/2
        A = (A >0).astype(int)
    else:
        A = pairwise_kernels(X, metric='rbf', gamma=1)
    return A


def add_multiple_layers(list_g,n_top_attrs = 2,knn_featrs = True,num_knn = 3):
    for g in list_g:
        g.knn_g = None
        g.edge_mat_knn = None
        g.attr_gs = []
        g.edge_mat_attrs = []
        if(knn_featrs):
            A = build_kneighbors(g.node_features,n_neighbors = num_knn)
            g_knn = nx.convert_matrix.from_numpy_matrix(A)
            g.knn_g = g_knn
            g.edge_mat_knn = get_edges_mat(g.knn_g)
        if(n_top_attrs>0):
            c = Counter(g.node_tags)
            top_vals = c.most_common(n_top_attrs)
            for value, count in top_vals:
                G_cur = nx.Graph()
                G_cur.add_nodes_from(g.g.nodes())
                imp_nodes = [node for i,node in enumerate(g.g.nodes()) if (g.node_tags[i] == value)]
                if(len(imp_nodes)<=1):
                    break
                for u,v in itertools.combinations(imp_nodes, 2):
                    G_cur.add_edge(u,v)
                g.attr_gs.append(G_cur)
                g.edge_mat_attrs.append(get_edges_mat(G_cur))
        

def load_data(dataset, degree_as_tag, root_folder = ".."):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/dataset/%s/%s.txt' % (root_folder, dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    max_nodes = 0
    min_nodes = 1e8
    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        if(g.node_features.shape[0]>max_nodes):
            max_nodes = g.node_features.shape[0]
        if(g.node_features.shape[0]<min_nodes):
            min_nodes = g.node_features.shape[0]
    
    print('# classes: %d' % len(label_dict))
    print('# feature_size: %d' % g_list[0].node_features.shape[1])
    print('# max nodes : %d' % max_nodes)
    print('# min nodes : %d' % min_nodes)
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

'''
tu dataset loading  funcs start
'''
def get_onehot_features(graph_info,max_node_tag):
    feature_size = max_node_tag

    node_feats = []
    #print(graph_info.ndata["node_labels"])
    for node in range(graph_info.ndata["node_labels"].shape[0]):
        cur_feat = np.zeros(feature_size,dtype = np.float32)
        #print(graph_info.ndata["node_labels"][node])
        cur_label = graph_info.ndata["node_labels"][node].item()
        cur_feat[cur_label] = 1
        assert cur_feat.nonzero()[0] == cur_label
        node_feats.append(cur_feat)
    
    return np.stack(node_feats,axis = 0)


def make_edge_mat_neighs(g_svg):
        #add labels and edge_mat       
    g = g_svg
    g.neighbors = [[] for i in range(len(g.g))]
    for i, j in g.g.edges():
        g.neighbors[i].append(j)
        g.neighbors[j].append(i)
    degree_list = []
    for i in range(len(g.g)):
        g.neighbors[i] = g.neighbors[i]
        degree_list.append(len(g.neighbors[i]))
    g.max_neighbor = max(degree_list)
    
    

    edges = [list(pair) for pair in g.g.edges()]
    edges.extend([[i, j] for j, i in edges])

    deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))

    
    

def load_tu_dataset(dataset_name):
    data_dgl = TUDataset(dataset_name)
    node_labels = False
    if("node_labels" in data_dgl.__dict__["attr_dict"].keys()):
        node_labels = True
    if(node_labels):
        num_labels = data_dgl.num_labels[0]
        max_node_tag = 0
        for g_info in data_dgl.__dict__['graph_lists']:
            cur_tag= g_info.ndata['node_labels'].max().item()+1
            if(cur_tag > max_node_tag ):
                max_node_tag = cur_tag
                #print(cur_tag)

    
    g_list = []
    max_nodes = 0
    min_nodes = 1000000
    for g_l , g_info in zip(data_dgl.graph_labels, data_dgl.graph_lists):
        node_feats = None
        node_tags = None
        g_nx = g_info.to_networkx()
        
        if(node_labels):
            #print(g_info.__dict__)
            node_feats = get_onehot_features(g_info, max_node_tag)
            node_tags = g_info.ndata["node_labels"].numpy().flatten()
            
        if("node_attr" in g_info.ndata.keys()):
            attrs = g_info.ndata["node_attr"].numpy().astype(np.float32)
            node_feats = np.concatenate([node_feats,attrs],axis = 1)
        l = g_l[0]
        g_svg = S2VGraph(g_nx, l, node_tags)
        g_svg.node_features = node_feats
        #g_svg.neighbors = l_neighs
        make_edge_mat_neighs(g_svg)
        #print(g_svg.__dict__)
        
        g_list.append(g_svg)
        if(g_svg.node_features.shape[0]>max_nodes):
            max_nodes = g_svg.node_features.shape[0]
        if(g_svg.node_features.shape[0]<min_nodes):
            min_nodes = g_svg.node_features.shape[0]
    
    print('# classes: %d' % data_dgl.num_labels[0].item())
    print('# feature_size: %d' % g_list[0].node_features.shape[1])
    print('# max nodes : %d' % max_nodes)
    print('# min nodes : %d' % min_nodes)
    print("# data: %d" % len(g_list))
    return g_list, data_dgl.num_labels[0].item()

'''

tu dataset loading finish 

'''
def separate_data(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

"""Get indexes of train and test sets"""
def separate_data_idx(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx