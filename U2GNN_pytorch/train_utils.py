
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time
from .pytorch_U2GNN_UnSup import TransformerU2GNN
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from .util import load_data, separate_data_idx
from sklearn.linear_model import LogisticRegression
import statistics

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        
def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl

def get_graphpool(batch_graph,args):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(args.device)

def get_idx_nodes(graph_indices, args,selected_graph_idx):
    idx_nodes = [torch.where(graph_indices==i)[0] for i in selected_graph_idx]
    idx_nodes = torch.cat(idx_nodes)
    return idx_nodes.to(args.device)

def get_batch_data(graphs,graph_indices,selected_idx,args):
    batch_graph = [graphs[idx] for idx in selected_idx]

    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size) #[1,1,1,1]
        X_concat = X_concat * 0.01
    X_concat = torch.from_numpy(X_concat).to(args.device)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    input_neighbors = []
    for input_node in range(X_concat.shape[0]):
        if input_node in dict_Adj_block:
            input_neighbors.append([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors + 1)])
    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).to(args.device)

    input_y = get_idx_nodes(graph_indices, args,selected_idx)

    return X_concat, input_x, input_y

class Batch_Loader(object):
    def __init__(self,graphs,graph_indices,args):
        self.graphs=graphs
        self.args=args
        self.graph_indices=graph_indices
    def __call__(self):
        selected_idx = np.random.permutation(len(self.graphs))[:self.args.batch_size]
        X_concat, input_x, input_y = get_batch_data(self.graphs, self.graph_indices, selected_idx,self.args)
        return X_concat, input_x, input_y
    



def data_loading_util(args):
    # Load data
    print("Loading data...")
    use_degree_as_tag = False
    if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI':
        use_degree_as_tag = True
    graphs, num_classes = load_data(args.dataset, use_degree_as_tag)
    graph_labels = np.array([graph.label for graph in graphs])
    feature_dim_size = graphs[0].node_features.shape[1]
    print(feature_dim_size)
    if "REDDIT" in args.dataset:
        feature_dim_size = 4
    args.update(feature_dim_size=feature_dim_size)
    args.update(num_graphs=len(graphs))
    
    graph_pool = get_graphpool(graphs,args)
    graph_indices = graph_pool._indices()[0]
    vocab_size=graph_pool.size()[1]
    batch_nodes = Batch_Loader(graphs,graph_indices,args)
    args.update(vocab_size=vocab_size)
    data_args= {}
    data_args['graphs']=graphs
    data_args['graph_labels']=graph_labels
    data_args['graph_pool'] = graph_pool
    data_args['graph_indices'] = graph_indices
    data_args['batch_nodes'] = batch_nodes
    data_args = Namespace(**data_args)
    print("Loading data... finished!")
    return data_args, args

def model_creation_util(parameterization,args):
    print(args.feature_dim_size)
    model = TransformerU2GNN(feature_dim_size=args.feature_dim_size, ff_hidden_size=parameterization['ff_hidden_size'],
                        dropout=parameterization['dropout'], num_self_att_layers=parameterization['num_timesteps'],
                        vocab_size=args.vocab_size, sampled_num=parameterization['sampled_num'],
                        num_U2GNN_layers=parameterization['num_hidden_layers'], device=args.device).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameterization['learning_rate'])
    num_batches_per_epoch = int((args.num_graphs- 1) / args.batch_size) + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)
    model_args = {'model':model, 'optimizer':optimizer, 'num_batches_per_epoch':num_batches_per_epoch, 'scheduler':scheduler}
    
    return Namespace(**model_args)

def single_epoch_training_util(data_args, model_args):
    model_args.model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(model_args.num_batches_per_epoch):
        X_concat, input_x, input_y = data_args.batch_nodes()
        model_args.optimizer.zero_grad()
        logits = model_args.model(X_concat, input_x, input_y)
        loss = torch.sum(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_args.model.parameters(), 0.5)
        model_args.optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(epoch, data_args, model_args):
    model = model_args.model
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.ss.weight
        graph_embeddings = torch.spmm(data_args.graph_pool, node_embeddings).data.cpu().numpy()
        acc_10folds = []
        for fold_idx in range(10):
            train_idx, test_idx = separate_data_idx(data_args.graphs, fold_idx)
            train_graph_embeddings = graph_embeddings[train_idx]
            test_graph_embeddings = graph_embeddings[test_idx]
            train_labels = data_args.graph_labels[train_idx]
            test_labels = data_args.graph_labels[test_idx]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_graph_embeddings, train_labels)
            ACC = cls.score(test_graph_embeddings, test_labels)
            acc_10folds.append(ACC)
            print('epoch ', epoch, ' fold ', fold_idx, ' acc ', ACC)

        mean_10folds = statistics.mean(acc_10folds)
        std_10folds = statistics.stdev(acc_10folds)
        # print('epoch ', epoch, ' mean: ', str(mean_10folds), ' std: ', str(std_10folds))

    return mean_10folds, std_10folds

def train_evaluate(data_args,model_args,args):
    cost_loss = []
    mean_10folds_best = -1
    std_10folds_best = -1
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        train_loss = single_epoch_training_util(data_args, model_args)
        cost_loss.append(train_loss)
        mean_10folds, std_10folds = evaluate(epoch, data_args, model_args)
        print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | mean {:5.2f} | std {:5.2f} | '.format(
                    epoch, (time.time() - epoch_start_time), train_loss, mean_10folds*100, std_10folds*100))

        if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
            model_args.scheduler.step()
        if(mean_10folds>mean_10folds_best):
            
            mean_10folds_best = mean_10folds
            std_10folds_best = std_10folds
    return mean_10folds_best, std_10folds_best


