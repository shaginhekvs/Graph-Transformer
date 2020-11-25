#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from pytorch_mlusgt_UnSup import TransformerMLUSGT
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
from sklearn.linear_model import LogisticRegression
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument('--sampled_num', default=512, type=int, help='')
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--model_siamese", default=False, type=bool, help="")

parser.add_argument("--num_knn", default=2, type=int, help="")
parser.add_argument("--use_l_att", default=False, type=bool, help="")
parser.add_argument("--num_attr_layers", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Timestep T ~ Number of self-attention layers within each U2GNN layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")
add_knn_layer = False
if(args.num_knn>0):
    add_knn_layer = True

use_degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI':
    use_degree_as_tag = True
graphs, num_classes = load_data(args.dataset, use_degree_as_tag)
add_multiple_layers(graphs,n_top_attrs = args.num_attr_layers,knn_featrs = add_knn_layer,num_knn = args.num_knn)

num_graph_layers = 1
if(args.num_attr_layers >0):
    num_graph_layers += args.num_attr_layers
if(args.num_knn > 0 ):
    num_graph_layers += 1

graph_labels = np.array([graph.label for graph in graphs])
feature_dim_size = graphs[0].node_features.shape[1]
print(feature_dim_size)
if "REDDIT" in args.dataset:
    feature_dim_size = 4

def get_Adj_matrix(batch_graph,mat_type = 'base', edge_mat_idx = 0):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat = None
        if(mat_type == 'base'):
            edge_mat = graph.edge_mat
        elif(mat_type == 'knn'):
            edge_mat = graph.edge_mat_knn
        elif(mat_type == "attrs"):
            edge_mat = graph.edge_mat_attrs[edge_mat_idx]
        else:
            raise ValueError("unknown edge mat string {}".format(mat_type))
        edge_mat_list.append(edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl

def get_graphpool(batch_graph):
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

    return graph_pool.to(device)
#
graph_pool = get_graphpool(graphs)
graph_indices = graph_pool._indices()[0]
vocab_size=graph_pool.size()[1]

def get_idx_nodes(selected_graph_idx):
    idx_nodes = [torch.where(graph_indices==i)[0] for i in selected_graph_idx]
    idx_nodes = torch.cat(idx_nodes)
    return idx_nodes.to(device)


def get_single_layer_neighbors(batch_graph, mat_type, idx_type, num_nodes):
    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph,mat_type =mat_type, edge_mat_idx = idx_type)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    input_neighbors = []
    for input_node in range(num_nodes):
        if input_node in dict_Adj_block:
            input_neighbors.append([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors + 1)])
    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).to(device)
    return input_x
    
def get_neighbors(batch_graph, num_nodes):
    mat_type = ["base"]
    idx_type = [[0]]
    if(args.num_knn > 0):
        mat_type.append("knn")
        idx_type.append([0])
    if(args.num_attr_layers > 0):
        mat_type.append("attrs")
        idx_type.append(list(range(0,args.num_attr_layers)))
    neighbors = []
    for list_idx, mat_name in zip(idx_type,mat_type):
        for idx in list_idx:
            #print(idx)
            neighbors.append(get_single_layer_neighbors(batch_graph,mat_type = mat_name, idx_type = idx,num_nodes = num_nodes))
    return neighbors
            
        
    
    
def get_batch_data(selected_idx):
    batch_graph = [graphs[idx] for idx in selected_idx]

    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    num_nodes = X_concat.shape[0]
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size) #[1,1,1,1]
        X_concat = X_concat * 0.01
    X_concat = torch.from_numpy(X_concat).to(device)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    
    neighbors = get_neighbors(batch_graph, num_nodes)
    input_x = torch.stack(neighbors,axis = 0)
    X_concat = torch.stack([X_concat]*num_graph_layers,axis = 0)
    #print(input_x.shape)
    #print(X_concat.shape)
    input_y = get_idx_nodes(selected_idx)
    #print(input_y)
    return X_concat, input_x, input_y

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(graphs))[:args.batch_size]
        X_concat, input_x, input_y = get_batch_data(selected_idx)
        return X_concat, input_x, input_y

batch_nodes = Batch_Loader()



print("Loading data... finished!")


model = TransformerMLUSGT(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        dropout=args.dropout, num_self_att_layers=args.num_timesteps,
                        vocab_size=vocab_size, sampled_num=args.sampled_num,
                        num_U2GNN_layers=args.num_hidden_layers, device=device, 
                          l_att = args.use_l_att , num_graph_layers=num_graph_layers,siamese = args.model_siamese).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(num_batches_per_epoch):
        X_concat, input_x, input_y = batch_nodes()
        optimizer.zero_grad()
        logits = model(X_concat, input_x, input_y)
        loss = torch.sum(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate():
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.ss.weight
        graph_embeddings = torch.spmm(graph_pool, node_embeddings).data.cpu().numpy()
        acc_10folds = []
        for fold_idx in range(10):
            train_idx, test_idx = separate_data_idx(graphs, fold_idx)
            train_graph_embeddings = graph_embeddings[train_idx]
            test_graph_embeddings = graph_embeddings[test_idx]
            train_labels = graph_labels[train_idx]
            test_labels = graph_labels[test_idx]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_graph_embeddings, train_labels)
            ACC = cls.score(test_graph_embeddings, test_labels)
            acc_10folds.append(ACC)
            print('epoch ', epoch, ' fold ', fold_idx, ' acc ', ACC)

        mean_10folds = statistics.mean(acc_10folds)
        std_10folds = statistics.stdev(acc_10folds)
        # print('epoch ', epoch, ' mean: ', str(mean_10folds), ' std: ', str(std_10folds))

    return mean_10folds, std_10folds

"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_U2GNN_UnSup", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    mean_10folds, std_10folds = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | mean {:5.2f} | std {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, mean_10folds*100, std_10folds*100))

    if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
        scheduler.step()

    write_acc.write('epoch ' + str(epoch) + ' mean: ' + str(mean_10folds*100) + ' std: ' + str(std_10folds*100) + '\n')

write_acc.close()
