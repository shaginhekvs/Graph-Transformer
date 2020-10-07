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
import math
import numpy.linalg as lg
import scipy.io
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



def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def accuracy_clustering(y_true, y_pred):
    
    # Ordering labels
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    
    scores = []
    
    # Try all the possible permutations
    permutations = list(itertools.permutations(labels))
    for perm in permutations:
        y_permuted = np.zeros_like(y_true)
        for i,k in enumerate(perm):
            y_permuted[y_pred==k] = labels[i]
        score = accuracy_score(y_true, y_permuted)
        scores.append(score)
    
    return max(scores)

def print_evaluation(y_true, L, K):

    _, V   = scipy.linalg.eigh(L)
    E      = V[:,:K]
    y = KMeans(K, random_state=42).fit_predict(E)

    acc_spec = accuracy_clustering(y_true, y)
    pu_spec        = purity_score(y_true, y)
    nmi_score_spec = nmi(y_true.ravel(), y.ravel())#, average_method='geometric')
    ri_score_spec  = ri(y_true.ravel(), y.ravel())

    print('Accuracy', acc_spec, 'Purity', pu_spec, 'NMI', nmi_score_spec, 'RI', ri_score_spec)
    return y


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
        laplacian[:,:,i] = sgwt_raw_laplacian(adjancency[:,:,i])
        
        if i==0:
            signal = aux.transpose([1,0])
        else:                           
            signal = np.concatenate((signal, aux.transpose([1,0])), axis=1)

    y_true = labels.ravel()  
    K      = len(np.unique(y_true))
    
    n_neighbors = 5
    
    return adjancency, laplacian, signal, y_true, K, n_neighbors


def sgwt_raw_laplacian(B):
    B         = B.T;
    N         = B.shape[0] 
    degrees   = B.sum(axis=1)
    diagw     = np.diag(B)

    nj2,ni2   = B.nonzero() 
    w2        = np.extract(B!=0,B)
    ndind     = (ni2!=nj2).nonzero()
    ni        = ni2[ndind]
    nj        = nj2[ndind]
    w         = w2[ndind]
    di        = np.arange(0,N)
    #dL        = 1 - diagw / degrees       
    #dL[degrees==0] = 0
    #ndL       = -w / (np.sqrt(degrees[ni]*degrees[ni])).flatten() 
    L         = csr_matrix((np.hstack((-w,degrees-diagw)), (np.hstack((ni,di)), np.hstack((nj,di)))), shape=(N, N)).toarray()

    return L

def build_kneighbors(X, knn=True, n_neighbors=20):
    if knn:
        A = kneighbors_graph(X, n_neighbors, include_self=False)
        A = np.array(A.todense())
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
            lap = sgwt_raw_laplacian(lap)
            #adj = - lap.copy()
            #np.fill_diagonal(adj, 0)
            
            L[:,:,i] = lap
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
    
    return L, y_true, K, L.shape[0], L.shape[2], X, W

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


def sgwt_raw_laplacian(B):
    B         = B.T;
    N         = B.shape[0] 
    degrees   = B.sum(axis=1)
    diagw     = np.diag(B)

    nj2,ni2   = B.nonzero() 
    w2        = np.extract(B!=0,B)
    ndind     = (ni2!=nj2).nonzero()
    ni        = ni2[ndind]
    nj        = nj2[ndind]
    w         = w2[ndind]
    di        = np.arange(0,N)
    #dL        = 1 - diagw / degrees       
    #dL[degrees==0] = 0
    #ndL       = -w / (np.sqrt(degrees[ni]*degrees[ni])).flatten() 
    L         = csr_matrix((np.hstack((-w,degrees-diagw)), (np.hstack((ni,di)), np.hstack((nj,di)))), shape=(N, N)).toarray()

    return L
def generate_train_test_mask(num_nodes, fraction = 0.6):
    n = num_nodes
    train_mask = np.zeros(n,dtype = bool)
    random_indices = np.random.permutation(range(n))
    train_indices = random_indices[:int(fraction*n)]
    train_mask[train_indices] = True
    test_mask = np.zeros(n,dtype = bool)
    test_indices = random_indices[int(fraction*n):]
    test_mask[test_indices]= True
    return train_mask, test_mask

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
    adj_mats = []
    Ls = []
    sum_ = 0
    for layer in layers : 
        df = edges_df[edges_df['layerId'] == layer]
        G= nx.from_pandas_edgelist(df, source='src', target='dst',create_using = nx.DiGraph)
        graphs.append(G)
        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
        
        adj_mats.append(adj_mat)
      
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
    return graphs, final_random_X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L ,adj

def add_edges_for_index(df, index_this, layer_id, G, col_prefix = "vote"):
    index_vote = df.iloc[index_this].loc["{}{}".format(col_prefix, layer_id)]
    if(index_vote == "?"):
        #print(index_vote)
        return []
        
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
    edges_df.loc[edges_df['layerId'] == "republican",'labels'] = 1 
    ids = np.array(list(range(len(edges_df))))
    graphs_list = []
    Ls = []
    adj_mats = []
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
        
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    
    print("# edges are {}".format( sum_))
    
    n = len(edges_df)
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n,args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    random_X = np.random.normal(size = [n, size_x])
    final_random_X = np.stack([random_X]* len(layer_ids),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    labels = np.array(list(edges_df['labels']))
    L = np.stack(Ls,axis = 2)
    final_random_X = torch.from_numpy(final_random_X).float()
    return graphs_list, final_random_X , torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask),L, adj
    
def get_mammo_dataset(args):
    multiplex_folder_path = args.multiplex_folder_path
    size_x = args.size_x
    mammo_data_folder = os.path.join(multiplex_folder_path, "Mammogram Dataset" )
    edges_file_path = os.path.join(mammo_data_folder,"mammographic_masses.data")
    layer_ids = list(range(0,5))
    layer_names= ["layer{}".format(i) for i in layer_ids]
    edges_df = pd.read_csv(edges_file_path, sep = ",", header = None, names =  layer_names + ["labels"]  )
    
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
        Ls.append(sgwt_raw_laplacian(adj_mat))
        graphs_list.append(G)
        adj_mats.append(np.array(adj_mat,dtype=int))
        
        sum_ += adj_mat.sum()
        print("# edges in layer {} are {}".format( layer, adj_mat.sum()))
    
    print("# edges are {}".format( sum_))
    
    n = len(edges_df)
    print("# nodes are {}".format( n ))
    train_mask, test_mask = generate_train_test_mask(n,args.train_fraction)
    print("# train samples are {}".format(train_mask.sum()))
    print("# test samples are {}".format(test_mask.sum()))
    X = edges_df.iloc[ids].loc[:,layer_names].replace("?", -1).to_numpy().astype(float)
    X = preprocessing.scale(X)
    #random_X = np.random.normal(size = [n, size_x])
    #final_random_X = np.stack([random_X]* len(layer_ids),axis = 2)
    adj = np.stack(adj_mats, axis = 2)
    L = np.stack(Ls,axis = 2)
    labels = np.array(list(edges_df.iloc[ids]['labels'])).astype(int)
    X = torch.from_numpy(X).float()
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
    adj_mats = []
    Ls = []
    sum_ = 0
    for layer in layer_ids:
        G = nx.DiGraph()
        G.add_nodes_from(ids)
        for i in ids:
            add_edges_for_index(edges_df, i, layer, G, col_prefix="layer")

        adj_mat = np.array(nx.adjacency_matrix(G).todense(),dtype=int)
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
    file_names = ["LN_2000_2004.mat", "LN_2005_2009.mat" , "LN_2010_2014.mat"]
    adj_mats = []
    G = []
    Ls = []
    sum_ = 0
    for i, file  in enumerate(file_names):
        
        mat1 = scipy.io.loadmat( os.path.join(data_folder, file))
        
        adj = process_adj_mat(mat1["A{}".format(i+2)])
        Ls.append(sgwt_raw_laplacian(adj))
        adj_mats.append(adj)
        print("# edges in layer {} are {}".format( i + 1, adj.sum()))
        sum_ += adj.sum()
        G.append(nx.convert_matrix.from_numpy_array(adj, create_using = nx.DiGraph))
    labels_mat = scipy.io.loadmat( os.path.join(data_folder, "LN_true.mat"))
    labels= np.array(labels_mat["s_LNG"].flatten(), dtype = int)
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
    return G, final_random_X, torch.from_numpy(labels),  torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask),L, adj
    

def generate_synthetic_dataset(n=200,K=5, sparse = False, size_x = 8, graph_type = "gaussian",ng_path = None):

    L, labels, K, n, S, X, adj = build_multilayer_graph(graph_type = graph_type, n=n, K=K, 
                                                show_graph=True, seed_nb = 100, ng_path = ng_path)

    feats = np.eye(n,dtype = np.float64)
    G_array = []
    for i in range(adj.shape[-1]):
        G_array.append(nx.from_numpy_array(adj[:,:,i]))
    train_mask, test_mask  = generate_train_test_mask(n)
    
    if(sparse):
        X = numpy_to_sparse(sklearn.preprocessing.scale(X[0]))
        X = Variable(make_sparse(X[0]))
        adj = numpy_to_sparse(adj[:,:,0])
    
    else:
        #X = Variable(torch.from_numpy(sklearn.preprocessing.scale(X[0])).float())
        #X = Variable(torch.from_numpy(feats).float())
        #X = torch.from_numpy(feats).float()
        if(graph_type == "gaussian"):
            print(X.shape)
            X = np.moveaxis(X, 0, 2)
            #X = np.moveaxis(X,0,1)
            #X = X.permute(1,2,0)
            print(X.shape)
            random_X = np.random.normal(size = [n,size_x])
            final_random_X = np.stack( [random_X]* 4,axis = 2)
            print(final_random_X.shape)
            X = np.concatenate([X, final_random_X] , axis = 1)
            print(X.shape)
            
        else:
            X = PCA(n_components=size_x).fit_transform(X)
            X = np.stack([X] * 3, axis = 2)
            print(X.shape)
        
        X = torch.from_numpy(X).float()
        adj = adj

    return G_array, X , torch.from_numpy(labels).int(), torch.from_numpy(train_mask), torch.from_numpy(test_mask), torch.from_numpy(test_mask), L, adj