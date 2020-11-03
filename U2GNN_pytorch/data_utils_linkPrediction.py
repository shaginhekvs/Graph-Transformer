import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import scipy
import pickle
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import pandas as pd
import seaborn as sns
import h5py
import math
import numpy.linalg as lg
from scipy import sparse
import scipy.linalg as slg
import torch
import sklearn
from torch.autograd import Variable
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ri
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, SpectralClustering
import itertools
from scipy.sparse import csr_matrix
import scipy.io 



def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v))




def read_data(name, path):
    data_file = scipy.io.loadmat(path)
    
    input_data = data_file['data']
    labels     = (data_file['truelabel'][0][0])
    
    nb_view    = len(input_data[0])
    N          = labels.ravel().shape[0]
    
    adjancency = np.zeros((N, N, nb_view))
    laplacian  = np.zeros((N, N, nb_view))
    
    for i in range(nb_view): 
        aux = input_data[0][i]
        if type(aux) is scipy.sparse.csc.csc_matrix:
            aux = aux.toarray()
        adjancency[:,:,i] = build_kneighbors(aux.transpose([1,0]), n_neighbors=5)
        
        if i==0:
            signal = aux.transpose([1,0])
        else:                           
            signal = np.concatenate((signal, aux.transpose([1,0])), axis=1)

    y_true = labels.ravel()  
    K      = len(np.unique(y_true))
    
    n_neighbors = 5
    
    return adjancency, signal, y_true, K, n_neighbors



def build_kneighbors(X, knn=True, n_neighbors=20):
    if knn:
        A = kneighbors_graph(X, n_neighbors, include_self=False)
        #A = np.array(A.todense())
        A = (A + A.T)/2
        A = (A >0).astype(int)
    else:
        A = pairwise_kernels(X, metric='rbf', gamma=1)
    return A


def draw_features(n_samples, n_dims, n_clusters, mean_scale, cov_scale, num=5):
    
    clusters = []
    for i in range(n_clusters):
        mean = mean_scale * np.random.randn(n_dims)
        cov = 0
        for i in range(num):
            cov_mat = cov_scale/num * np.random.randn(n_dims, n_dims)
            cov = cov + cov_mat.T @ cov_mat
        X = np.random.multivariate_normal(mean, cov, n_samples)
        clusters.append(X)
    return tuple(clusters)


def build_multilayer_graph(graph_type = 'gaussian', n=50, K=5, show_graph=True, seed_nb = 50, ng_path = None):
    # n: total number of nodes
    # m: nb of clusters 
    # signals: dimension of signals 
    
    # generate a graph
                
    y_true = None
    X = None 
               
    if graph_type =='gaussian':
    
        np.random.seed(seed_nb)

        mean_scale = 3
        cov_scale = 3

        X11, X12, X13, X14, X15 = draw_features(int(n/K), 2, K, mean_scale, cov_scale)
        X21, X22, X23, X24, X25 = draw_features(int(n/K), 2, K, mean_scale, cov_scale)
        X31, X32, X33, X34, X35 = draw_features(int(n/K), 2, K, mean_scale, cov_scale)
        X41, X42, X43, X44, X45 = draw_features(int(n/K), 2, K, mean_scale, cov_scale)

        sig1 = np.concatenate([X11,X12,X13,X14,X15], axis=0)
        sig2 = np.concatenate([X21,X22,X23,X24,X25], axis=0)
        sig3 = np.concatenate([X31,X32,X33,X34,X35], axis=0)
        sig4 = np.concatenate([X41,X42,X43,X34,X45], axis=0)
        signals  = np.stack([sig1, sig2, sig3, sig4], axis=0)
        X = signals/np.max(signals)

        y_true = np.zeros(n)
        Nodes = int(n/K)
        y_true[       :1*Nodes] = 0
        y_true[1*Nodes:2*Nodes] = 1
        y_true[2*Nodes:3*Nodes] = 2
        y_true[3*Nodes:4*Nodes] = 3
        y_true[4*Nodes:5*Nodes] = 4

        # Graph construction
        L = np.zeros((n,n,4))
        W = np.zeros((n,n,4))
        for i in range(4):
            lap = build_kneighbors(signals[i], n_neighbors=10)
            adj = lap.copy()
            #lap = sgwt_raw_laplacian(lap)
            #adj = - lap.copy()
            #np.fill_diagonal(adj, 0)
            
            #L[:,:,i] = lap
            W[:,:,i] = adj
        
        if show_graph:
            plt.figure(figsize=(15,3))

            alpha = 0.4
            markers = ['o', 's', '^', 'X', '*']
            size = 10

            plt.subplot(1,4,1)
            for i, data in enumerate([X11, X12, X13, X14, X15]):
                plt.plot(data[:,0], data[:,1], markers[i], alpha=alpha, ms=size, mew=2)

            plt.subplot(1,4,2)
            for i, data in enumerate([X21, X22, X23, X24, X25]):
                plt.plot(data[:,0], data[:,1], markers[i], alpha=alpha, ms=size, mew=2)

            plt.subplot(1,4,3)
            for i, data in enumerate([X31, X32, X33, X34, X35]):
                plt.plot(data[:,0], data[:,1], markers[i], alpha=alpha, ms=size, mew=2)

            plt.subplot(1,4,4)
            for i, data in enumerate([X41, X42, X43, X44, X45]):
                plt.plot(data[:,0], data[:,1], markers[i], alpha=alpha, ms=size, mew=2)
                
    if graph_type in ['NGs']:
        W, L, X, y_true, K, _ = read_data(graph_type, ng_path)
    
    return y_true, K, L.shape[0], L.shape[2], X, W

def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def numpy_to_sparse(ary):
    return sparse.csr_matrix(ary)
    
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return adj_normalized



def get_vicker_chan_dataset(args):
    multiplex_folder_path = args.multiplex_folder_path
    size_x = args.size_x
    vicker_data_folder = os.path.join(multiplex_folder_path, "Vickers-Chan Dataset" , "Dataset")
    edges_file_path = os.path.join(vicker_data_folder,"Vickers-Chan-7thGraders_multiplex.edges" )
    edges_df = pd.read_csv(edges_file_path, sep = " ", header = None,  names = ["layerId", "src", "dst", "weight"],dtype=int)
    edges_df['src'] = edges_df['src'] - 1 # index IDs from 0
    edges_df['dst'] = edges_df['dst'] - 1 # index IDs from 0
    layers = [1, 2, 3]
    graphs = []
    edges = []
    adj_mats = []
    Ls = []
    sum_ = 0
    for layer in layers : 
        df = edges_df[edges_df['layerId'] == layer]
        G= nx.from_pandas_edgelist(df, source='src', target='dst',create_using = nx.DiGraph)
        graphs.append(G)
        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        
        adj_mats.append(adj_mat)
        idx_nonzeros = np.nonzero(adj_mat)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([layer,src,dst])
        Ls.append(sgwt_raw_laplacian(adj_mat))
        
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    print("# edges are {}".format( sum_))
    
    n = max(edges_df["src"]) + 1
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    random_X = np.random.normal(size = [n, size_x])
    final_random_X = np.stack([random_X]* len(layers),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    L = np.stack(Ls,axis = 2)
    labels = np.zeros(n,dtype = int) 
    labels[12:] = 1 # 0 for boy from index 0 - 11 , 12 - 28 is for girl
    final_random_X = torch.from_numpy(final_random_X).float()
    if(args.save_input_list):
        edges_np = np.array(edges,dtype=int)    
        np.savetxt(os.path.join(vicker_data_folder,"vicker_multiple_edges.txt"),edges_np,fmt='%i')
        np.savetxt(os.path.join(vicker_data_folder,"vicker_labels.txt"),labels,fmt='%i')
        print("saved to {}".format(vicker_data_folder))
    
    return graphs, final_random_X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L ,adj

def add_edges_for_index(df, index_this, layer_id, G, col_prefix = "vote"):
    index_vote = df.iloc[index_this].loc["{}{}".format(col_prefix, layer_id)]
    if(index_vote == "?"):
        pass
        #print(index_vote)
        #return []
        
    other_votes = [(index_this, val ) for val in list((df.loc[df["{}{}".format(col_prefix, layer_id)] == index_vote]).index)]
    #print(other_votes)
    G.add_edges_from(other_votes)
    return other_votes

def get_congress_dataset(args):
    multiplex_folder_path = args.multiplex_folder_path
    size_x = args.size_x
    vicker_data_folder = os.path.join(multiplex_folder_path, "Congress Dataset" )
    edges_file_path = os.path.join(vicker_data_folder,"house-votes-84.data")
    layer_ids = list(range(0,16))
    edges_df = pd.read_csv(edges_file_path, sep = ",", header = None,  names = ["layerId"] + ["vote{}".format(i) for i in layer_ids])
    edges_df['labels'] = 0
    #layer_ids = [0,1,2,3]
    edges_df.loc[edges_df['layerId'] == "republican",'labels'] = 1 
    ids = np.array(list(range(len(edges_df))))
    graphs_list = []
    n = len(edges_df)
    Ls = []
    adj_mats = []
    edges = []
    sum_ = 0
    for layer in layer_ids:
        G = nx.DiGraph()
        G.add_nodes_from(ids)
        for i in ids:
            add_edges_for_index(edges_df, i, layer, G)
        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        
        graphs_list.append(G)
        Ls.append(sgwt_raw_laplacian(adj_mat))
        adj_mats.append(np.array(adj_mat,dtype=int))
        idx_nonzeros = np.nonzero(adj_mat)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([layer,src,dst])
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    
    print("# edges are {}".format( sum_))
    edges_np = np.array(edges,dtype=int)    
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n,args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    random_X = np.random.normal(size = [n, size_x])
    final_random_X = np.stack([random_X]* len(layer_ids),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    labels = np.array(list(edges_df['labels']),dtype=int)
    L = np.stack(Ls,axis = 2)
    final_random_X = torch.from_numpy(final_random_X).float()
    if(args.save_input_list):
        np.savetxt(os.path.join(vicker_data_folder,"congress_multiple_edges.txt"),edges_np,fmt='%i')
        np.savetxt(os.path.join(vicker_data_folder,"congress_labels.txt"),labels,fmt='%i')
        print("saved to {}".format(vicker_data_folder))
    return graphs_list, final_random_X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask),L, adj
    
def get_mammo_dataset(args):
    multiplex_folder_path = args.multiplex_folder_path
    size_x = args.size_x
    mammo_data_folder = os.path.join(multiplex_folder_path, "Mammogram Dataset" )
    edges_file_path = os.path.join(mammo_data_folder,"mammographic_masses.data")
    layer_ids = list(range(0,5))
    layer_names= ["layer{}".format(i) for i in layer_ids]
    edges_df = pd.read_csv(edges_file_path, sep = ",", header = None, names =  layer_names + ["labels"]  )
    edges = []
    ids = np.array(list(range(len(edges_df))))
    graphs_list = []
    adj_mats = []
    Ls = []
    sum_ = 0
    for layer in layer_ids:
        G = nx.DiGraph()
        G.add_nodes_from(ids)
        for i in ids:
            add_edges_for_index(edges_df, i, layer, G, col_prefix="layer")

        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        idx_nonzeros = np.nonzero(adj_mat)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([layer,src,dst])
        Ls.append(sgwt_raw_laplacian(adj_mat))
        graphs_list.append(G)
        adj_mats.append(np.array(adj_mat,dtype=int))
        
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    
    print("# edges are {}".format( sum_))
    
    n = len(edges_df)
    print("# nodes are {}".format( n ))
    edges_np = np.array(edges,dtype=int)    
    train_mask, test_mask = generate_train_test_mask(n,args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    X = edges_df.iloc[ids].loc[:,layer_names].replace("?", -1).to_numpy().astype(float)
    X = preprocessing.scale(X)
    X = np.stack([X]* len(layer_ids),axis = 2)
    random_X = np.random.normal(size = [n,size_x])
    final_random_X = np.stack( [random_X]* len(layer_ids),axis = 2)
    X = np.concatenate([X, final_random_X] , axis = 1)
    adj = np.stack(adj_mats, axis = 2)
    L = np.stack(Ls,axis = 2)
    labels = np.array(list(edges_df.iloc[ids]['labels'])).astype(int)
    X = torch.from_numpy(X).float()
    if(args.save_input_list):
        np.savetxt(os.path.join(mammo_data_folder,"mammo_multiple_edges.txt"),edges_np,fmt='%i')
        np.savetxt(os.path.join(mammo_data_folder,"mammo_labels.txt"),labels,fmt='%i')
        print("saved to {}".format(mammo_data_folder))
    return graphs_list, X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L, adj
    
def get_balance_dataset(args):
    multiplex_folder_path = args.multiplex_folder_path
    size_x = args.size_x
    mammo_data_folder = os.path.join(multiplex_folder_path, "Balance-Scale Dataset" )
    edges_file_path = os.path.join(mammo_data_folder,"balance-scale.data")
    layer_ids = list(range(0,4))
    layer_names= ["layer{}".format(i) for i in layer_ids]
    edges_df = pd.read_csv(edges_file_path, sep = ",", header = None, names = ["labels"]+ layer_names   )
    print(edges_df.head())
    ids = np.array(list(range(len(edges_df))))
    graphs_list = []
    edges = []
    adj_mats = []
    Ls = []
    sum_ = 0
    for layer in layer_ids:
        G = nx.DiGraph()
        G.add_nodes_from(ids)
        for i in ids:
            add_edges_for_index(edges_df, i, layer, G, col_prefix="layer")

        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        idx_nonzeros = np.nonzero(adj_mat)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([layer,src,dst])
        Ls.append(sgwt_raw_laplacian(adj_mat))
        graphs_list.append(G)
        adj_mats.append(np.array(adj_mat))
        
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    
    print("# edges are {}".format( sum_))
    
    n = len(edges_df)
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    X = edges_df.iloc[ids].loc[:,layer_names].replace("?", -1).to_numpy().astype(float)
    X = preprocessing.scale(X)
    #random_X = np.random.normal(size = [n, size_x])
    #final_random_X = np.stack([random_X]* len(layer_ids),axis = 2)
    X = np.stack([X]* len(layer_ids),axis = 2)
    random_X = np.random.normal(size = [n,size_x])
    final_random_X = np.stack( [random_X]* 4,axis = 2)
    X = np.concatenate([X, final_random_X] , axis = 1)
    adj = np.stack(adj_mats, axis = 2)
    L = np.stack(Ls,axis = 2)
    edges_df["labels_style"] = edges_df["labels"].astype('category')
    labels = np.array(list(edges_df.iloc[ids]['labels_style'].cat.codes))
    X = torch.from_numpy(X).float()
    print(X.shape)
    if(args.save_input_list):
        edges_np = np.array(edges,dtype=int)    
        np.savetxt(os.path.join(mammo_data_folder,"balance_multiple_edges.txt"),edges_np,fmt='%i')
        np.savetxt(os.path.join(mammo_data_folder,"balance_labels.txt"),labels,fmt='%i')
        print("saved to {}".format(mammo_data_folder))
    
    return graphs_list, X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L, adj

 
def get_leskovec_dataset(args):
    multiplex_folder_path= args.multiplex_folder_path
    size_x = args.size_x
    les_data_folder = os.path.join(multiplex_folder_path, "Leskovec-Ng Dataset" )
    edges_file_path = os.path.join(les_data_folder,"Leskovec-Ng.multilayer.edges")
    labels = np.loadtxt(os.path.join(les_data_folder,'Leskovec-Ng.multilayer.labels')).astype(np.int32)
    
    data = np.loadtxt(fname=edges_file_path).astype(np.int32)
    layers = [0, 1, 2, 3]
    graphs = []
    adj_mats = []
    sum_ = 0
    Ls = []
    edges_df = pd.read_csv(edges_file_path, sep = " ", header = None,  names = ["layerId", "src", "dst"],dtype=int)
    print(edges_df['src'].min())
    
    for layer in layers : 
        df = edges_df[edges_df['layerId'] == layer]
        G= nx.from_pandas_edgelist(df, source='src', target='dst',create_using = nx.DiGraph)
        graphs.append(G)

        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        idx_nonzeros = np.nonzero(adj_mat)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([layer,src,dst])
        adj_mats.append(adj_mat)
        Ls.append(sgwt_raw_laplacian(adj_mat))
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    print("# edges are {}".format( sum_))
    
    n = max(edges_df["src"].max(), edges_df["dst"].max())  + 1
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    random_X = np.random.normal(size = [n, size_x])
    final_random_X = np.stack([random_X]* len(layers),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    L = np.stack(Ls,axis = 2)
    final_random_X = torch.from_numpy(final_random_X).float()

    return graphs, final_random_X, torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L, adj


def process_adj_mat(A):
    A[A>0] = 1
    return A.astype(int)

def get_leskovec_true_dataset(args):
    multiplex_folder_path= args.multiplex_folder_path
    size_x = args.size_x
    data_folder = os.path.join(multiplex_folder_path, "Leskovec-Ng Dataset" )
    file_names = ["LN_1995_1999.mat","LN_2000_2004.mat", "LN_2005_2009.mat" , "LN_2010_2014.mat"]
    adj_mats = []
    edges = []
    G = []
    Ls = []
    sum_ = 0
    for i, file  in enumerate(file_names):
        
        mat1 = scipy.io.loadmat( os.path.join(data_folder, file))
        
        adj = process_adj_mat(mat1["A{}".format(i+1)])
        Ls.append(sgwt_raw_laplacian(adj))
        adj_mats.append(adj)
        idx_nonzeros = np.nonzero(adj)
        for (src,dst) in zip(idx_nonzeros[0],idx_nonzeros[1]):
            edges.append([i,src,dst])
        print("# edges in layer {} are {}".format( i + 1, adj.sum()))
        sum_ += adj.sum()
        G.append(nx.convert_matrix.from_numpy_array(adj, create_using = nx.DiGraph))
    labels_mat = scipy.io.loadmat( os.path.join(data_folder, "LN_true.mat"))
    labels= np.array(labels_mat["s_LNG"].flatten(), dtype = int) - 1
    print("# edges are {}".format( sum_))
    n = adj_mats[0].shape[0]
    L = np.stack(Ls,axis = 2)
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    random_X = np.random.normal(size = [n, size_x])
    final_random_X = np.stack([random_X]* len(file_names),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    final_random_X = torch.from_numpy(final_random_X).float()
    print("# nodes are {}".format( n ))
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    if(args.save_input_list):
        edges_np = np.array(edges,dtype=int)    
        np.savetxt(os.path.join(data_folder,"leskovec_multiple_edges.txt"),edges_np,fmt='%i')
        np.savetxt(os.path.join(data_folder,"leskovec_labels.txt"),labels,fmt='%i')
        print("saved to {}".format(data_folder))
    return G, final_random_X, torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask),L, adj

def mat_file_load_all(fname) :
    f = h5py.File(fname)
    f_W = f['W']
    M = np.array(sparse.csc_matrix( (f_W['data'], f_W['ir'], f_W['jc']) ).todense())
    features = f[f['data'][0][0]].value
    labels = f[f["truelabel"][0][0]].value.squeeze()
    f.close()
    
    return M, features, labels

def load_ml_clustering_mat_dataset(args):
    data_folder = args.ml_cluster_mat_folder
    mat_file_path = os.path.join(data_folder, "{}.mat".format(args.dataset))
    adj, feats , labels = mat_file_load_all(mat_file_path)
    print("# edges in layer {} are {}".format( 1, adj.sum()))
    n = adj.shape[0]
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# nodes are {}".format( n ))
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    nx_g = nx.convert_matrix.from_numpy_array(adj, create_using = nx.DiGraph)
    nx_list = [nx_g]
    adj_list = [adj]
    Ls = [sgwt_raw_laplacian(adj)]
    if(args.scale_features):
        feats_scaled = sklearn.preprocessing.scale(feats)
    else:
        feats_scaled = feats
    if args.size_x < feats.shape[1] :
        features = torch.tensor(PCA(n_components=args.size_x).fit_transform(feats_scaled),dtype=torch.float).to(args.device)
    else:
        features = torch.tensor(feats_scaled,dtype=torch.float).to(args.device)
    print("# features are {}".format( features.shape[1]))
    features_list = [features]
    if(args.create_similarity_layer):
        adj_2 = np.array(kneighbors_graph(feats ,n_neighbors = args.num_similarity_neighbors, metric = "cosine",include_self = True).todense())
        nx_g2 = nx.convert_matrix.from_numpy_array(adj_2, create_using = nx.DiGraph)
        adj_list.append(adj_2)
        nx_list.append(nx_g2)
        features_list.append(features)
        Ls.append(sgwt_raw_laplacian(adj_2))
    
    adj_final = np.stack(adj_list,axis = 2)
    L = np.stack(Ls, axis = 2)
    features = torch.stack(features_list, axis = 2 ).to(args.device)

    
    return nx_list, features , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L ,adj_final
        
def load_ml_clustering_scipymat_dataset(args):
    data_folder = args.ml_cluster_mat_folder
    mat_file_path = os.path.join(data_folder, "{}.mat".format(args.dataset))
    print(mat_file_path)
    mat1 = scipy.io.loadmat( mat_file_path)
    num_layers = mat1['data'].shape[1]
    print("# num layers {}".format(num_layers))
    labels = mat1['truelabel'][0,0].squeeze()
    n = len(labels)
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# nodes are {}".format( n ))
    
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    feats_list = []
    nx_list = []
    adj_list = []
    Ls = []
    for i in range(num_layers):
        print("# current layer {}".format(i))
        feats = mat1['data'][0,i].T
        if(type(feats) == scipy.sparse.csr.csr_matrix or type(feats) == scipy.sparse.csc.csc_matrix):
            print("convert features to dense")
            feats = np.array(feats.todense())
        print(feats.shape)
        adj = np.array(kneighbors_graph(feats ,n_neighbors = args.num_similarity_neighbors, metric = "cosine",include_self = True).todense())
        print("# edges in layer {} are {}".format( i, adj.sum()))
        if(args.scale_features):
            feats_scaled = sklearn.preprocessing.scale(feats)
        else:
            feats_scaled = feats
        if args.size_x < feats.shape[1] :
            features = torch.tensor(PCA(n_components=args.size_x).fit_transform(feats_scaled),dtype=torch.float).to(args.device)
        elif args.size_x > feats.shape[1]:
            feats_scaled = sklearn.preprocessing.scale(feats)
            random_X = np.random.normal(size = [n, args.size_x - feats.shape[1]])
            feats_scaled = np.concatenate([feats_scaled,random_X], axis = 1)
            features = torch.tensor(feats_scaled,dtype=torch.float).to(args.device)
        else:
            features = torch.tensor(feats_scaled,dtype=torch.float).to(args.device)
        feats_list.append(features)
        nx_list.append(nx.convert_matrix.from_numpy_array(adj, create_using = nx.DiGraph))
        adj_list.append(adj)
        Ls.append(sgwt_raw_laplacian(adj))
        
    
    
    adj_final = np.stack(adj_list,axis = 2)
    L = np.stack(Ls, axis = 2)
    features = torch.stack(feats_list, axis = 2 ).to(args.device)
    print("# features are {}".format( features.shape[1]))

    
    return nx_list, features , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L ,adj_final

def get_uci_true_dataset(args):
    multiplex_folder_path= args.multiplex_folder_path

    data_folder = os.path.join(multiplex_folder_path, "UCI",  "mfeat")
    file_names = ["mfeat-fac" , "mfeat-fou", "mfeat-kar", "mfeat-mor" , "mfeat-pix" , "mfeat-zer"]
    adj_mats = []
    edges = []
    G = []
    Ls = []
    sum_ = 0
    labels = [[i] * 200 for i in range ( 10 )]
    np_labels = np.array(labels).flatten()
    n = len(np_labels)
    feats_list = []
    nx_list = []
    adj_list = []
    Ls = []
    for i, file  in enumerate(file_names):
        print(os.path.join(data_folder, file))
        print("# current layer {}".format(i))
        print(n)
        with open(os.path.join(data_folder, file),'r') as f:
            mat = f.readlines()
        #print(mat)
        mat_2d = [ l.split() for l in mat]
        np_ary = np.array(mat_2d, dtype = np.float)
        feats = np_ary
        adj = np.array(kneighbors_graph(feats ,n_neighbors = args.num_similarity_neighbors, metric = "cosine",include_self = True).todense())
        print("# edges in layer {} are {}".format( i, adj.sum()))
        if(args.scale_features):
            feats_scaled = sklearn.preprocessing.scale(feats)
        else:
            feats_scaled = feats
        if args.size_x < feats.shape[1] :
            features = torch.tensor(PCA(n_components=args.size_x).fit_transform(feats_scaled),dtype=torch.float).to(args.device)
        elif args.size_x > feats.shape[1]:
            feats_scaled = sklearn.preprocessing.scale(feats)
            random_X = np.random.normal(size = [n, args.size_x - feats.shape[1]])
            feats_scaled = np.concatenate([feats_scaled,random_X], axis = 1)
            features = torch.tensor(feats_scaled,dtype=torch.float).to(args.device)
        else:
            features = torch.tensor(feats_scaled,dtype=torch.float).to(args.device)
        feats_list.append(features)
        nx_list.append(nx.convert_matrix.from_numpy_array(adj, create_using = nx.DiGraph))
        adj_list.append(adj)
        Ls.append(sgwt_raw_laplacian(adj))
        
    
    train_mask, test_mask = generate_train_test_mask(n, args.train_fraction)
    print("# nodes are {}".format( n ))
    
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    adj_final = np.stack(adj_list,axis = 2)
    L = np.stack(Ls, axis = 2)
    features = torch.stack(feats_list, axis = 2 ).to(args.device)
    
        #mat1 = pd.read_csv( os.path.join(data_folder, file), sep = " ", header = None, ).to_numpy()
        #print(mat1.shape)
        
    return nx_list, features , torch.from_numpy(np_labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L ,adj_final

    
