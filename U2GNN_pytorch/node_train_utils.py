from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)
from dgl.data import CoraDataset, CitationGraphDataset, PPIDataset, KarateClub
import dgl
import numpy as np
np.random.seed(123)
import time
import networkx as nx
from .pytorch_U2GNN_UnSup import TransformerU2GNN
from .gat_pytorch import TransformerGAT
from .gcn_pytorch import TransformerGCN
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from .data_utils import generate_synthetic_dataset, get_vicker_chan_dataset, get_congress_dataset, get_mammo_dataset, get_balance_dataset, get_leskovec_dataset
from .util import load_data, separate_data_idx, Namespace
from sklearn.linear_model import LogisticRegression
import statistics

def process_adj_mat(adj,args):
    
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj.copy()
    #adj_label = Variable(torch.from_numpy(adj_label).float().to(args.device))
    adj_label = torch.from_numpy(adj_label).float().to(args.device)
    #adj_norm = Variable(torch.from_numpy(nx.normalized_laplacian_matrix(args.graph_obj).todense()).float().to(args.device))
    adj_norm = Variable(torch.from_numpy(adj.copy()).float().to(args.device))
    args.update(adj_norm = adj_norm)
    weight_mask = adj_label.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(args.device)
    weight_tensor[weight_mask] = pos_weight
    args.update(adj_label = adj_label)
    args.update(norm = torch.tensor(norm))
    args.update(weight_tensor = weight_tensor)
    

def get_input_generator(args):
      # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraDataset()
    elif args.dataset == 'citeseer':
        data = CitationGraphDataset('citeseet')
    elif args.dataset == 'pubmed':
        data = CitationGraphDataset('pubmed')
    elif args.dataset == 'PPIDataset':
        data = PPIDataset()
    elif args.dataset == "karate":
        data = KarateClub()
        g = data[0]
        g.ndata['feat'] = torch.eye(g.number_of_nodes()).to(args.device)
        g.ndata['train_mask'] = torch.from_numpy(np.ones((g.number_of_nodes(),), dtype=bool)).to(args.device)
        g.ndata['val_mask'] = torch.from_numpy(np.ones((g.number_of_nodes(),), dtype=bool)).to(args.device)
        g.ndata['test_mask'] = torch.from_numpy(np.ones((g.number_of_nodes(),), dtype=bool)).to(args.device)
    elif args.dataset == "synth":
        
        output = generate_synthetic_dataset()
        args.update(graph_obj = output[0])
        print(output[-1].shape)
        process_adj_mat(output[-1], args)
        return output[:-1]
    
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.device == 'cuda':
        cuda = False
    else:
        cuda = True
        #g = g.to(args.device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    print(sum(train_mask))
    val_mask = g.ndata['val_mask'].type(torch.BoolTensor)
    test_mask = g.ndata['test_mask'].type(torch.BoolTensor)
    if(not (args.dataset == "karate")):
        train_mask = ~ (val_mask | test_mask)
    print(sum(train_mask))
    num_feats = features.shape[1]
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # remove self loop
    #g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    nx_g = data[0].to_networkx()
    args.update(graph_obj = nx_g)
    adj = nx.convert_matrix.to_numpy_matrix(nx_g)
    process_adj_mat(adj, args)
    return nx_g, features, labels, train_mask, val_mask, test_mask


def get_batch_data_node(graphs, features, train_idx, args):
    '''
    returns:
    X_concat: concatenated features of all the selected graphs
    input_x: neighbor matrix #num_nodes X # num neighbours + 1 
    input_y: 1D Tensor of where the nodes of selected_graph are, in the sparse graph matrix.
    '''
    #X_concat = features[train_idx].to(args.device)
    input_neighbors = []
    for val in range(args.vocab_size):
        value = val
        neighbors_list = [n for n in graphs.neighbors(value)]
        if(neighbors_list):
            input_neighbors.append([value]+list(np.random.choice(neighbors_list, args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([value for _ in range(args.num_neighbors + 1)])

    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).to(args.device)
    input_y = torch.from_numpy(np.array([x for x in range(args.vocab_size)])).to(args.device)
    return features.to(args.device), input_x, input_y


class Batch_Loader_node_classification(object):
    def __init__(self,args):
        init_object = get_input_generator(args)
        self.graph = init_object[0]
        self.graph, self.features, self.label, self.train_mask, self.val_mask, self.test_mask = init_object
        self.args=args
    def __call__(self):
        train_idx = select_bs_indices_from_mask(self.train_mask, self.args.batch_size).to('cpu').numpy()
        
        X_concat, input_x, input_y = get_batch_data_node(self.graph, self.features, train_idx, self.args)
        return X_concat, input_x, input_y
    
    def get_validation_idx(self):
        return select_bs_indices_from_mask(self.test_mask,-1)
    
    def get_test_idx(self):
        return select_bs_indices_from_mask(self.test_mask,-1)
    
    def get_train_idx(self):
        return select_bs_indices_from_mask(self.train_mask,-1)



def select_bs_indices_from_mask(boolean_mask,bs):
    all_mask_idx = torch.where(boolean_mask==True)[0]
    if(bs>0):
        selected_idx = np.random.permutation(len(all_mask_idx))[:bs]
        return all_mask_idx[selected_idx]
    return all_mask_idx




def data_loading_util(args):
    # Load data
    print("Loading data...")
    
    batch_nodes = Batch_Loader_node_classification(args)
    args.update(vocab_size=batch_nodes.features.shape[0])
    args.update(trainset_size=sum(batch_nodes.train_mask).item())
    args.update(feature_dim_size=batch_nodes.features.shape[1])
    data_args= {}
    data_args['batch_nodes'] = batch_nodes
    data_args = Namespace(**data_args)
    print("Loading data... finished!")
    return data_args, args

def model_creation_util(parameterization,args):
    print(args.feature_dim_size)
    print(args.vocab_size)
    print("create model")
    print(args.model_type)
    args.update(sampler_type = "neighbor")
    model_input_args = dict(feature_dim_size=args.feature_dim_size, ff_hidden_size=parameterization['ff_hidden_size'],
                            dropout=parameterization['dropout'], num_self_att_layers=parameterization['num_timesteps'],
                            vocab_size=args.vocab_size, sampled_num=parameterization['sampled_num'],
                            num_U2GNN_layers=parameterization['num_hidden_layers'], device=args.device, sampler_type = args.sampler_type, graph_obj = args.graph_obj, loss_type = args.loss_type, adj_mat = args.adj_label)
    if(args.model_type == 'u2gnn'):
        model = TransformerU2GNN(**model_input_args).to(args.device)
    elif(args.model_type == 'gcn'):
        model = TransformerGCN(**model_input_args).to(args.device)
    elif (args.model_type == "gat"):
        model = TransformerGAT(**model_input_args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameterization['learning_rate'])
    else:
        raise ValueError(' {} isnt a valid model'.format(args.model_type))

    if(args.batch_size>0):
        num_batches_per_epoch = int((args.trainset_size- 1) // args.batch_size) + 1
    else:
        num_batches_per_epoch = 1
    print("model done")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)
    model_args = {'model':model, 'optimizer':optimizer, 'num_batches_per_epoch':num_batches_per_epoch, 'scheduler':scheduler}
    
    return Namespace(**model_args)

def loss_func(args, logits):
    if(args.loss_type == 'default'):
        embeds = logits[0]
        loss = torch.sum(embeds)
        
    elif(args.loss_type == 'gae'):
        embeds = logits[0]
        A_pred = torch.sigmoid(torch.matmul(embeds, embeds.t()))
        loss = args.norm*F.binary_cross_entropy(A_pred.view(-1), args.adj_label.view(-1), weight = args.weight_tensor)
    elif( args.loss_type == "contrastive"):
        loss = logits[0]
    return loss
        
def single_epoch_training_util(data_args, model_args, args):
    model_args.model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(model_args.num_batches_per_epoch):
        X_concat, input_x, input_y = data_args.batch_nodes()
        model_args.optimizer.zero_grad()
        logits = model_args.model(X_concat, input_x, input_y, args)
        print("forward pass done")
        loss = loss_func(args,logits)
        loss.backward()
        print("backward pass done")
        #torch.nn.utils.clip_grad_norm_(model_args.model.parameters(), 0.5)
        model_args.optimizer.step()
        total_loss += loss.item()

    return total_loss


def get_node_embeddings(data_args, model_args, args):
    model = model_args.model
    model.eval()
    if(args.loss_type == 'default'):
        
        return model.ss.weight.to('cpu')
        
    elif(args.loss_type == 'gae'):
        X_concat, input_x, input_y = data_args.batch_nodes()
        return model(X_concat, input_x, input_y, args)[1].detach().to('cpu')
    elif( args.loss_type == "contrastive"):
        X_concat, input_x, input_y = data_args.batch_nodes()
        return model(X_concat, input_x, input_y, args)[1].detach().to('cpu')

def evaluate(epoch, data_args, model_args, args):
    model = model_args.model
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = get_node_embeddings(data_args, model_args, args)
        acc_10folds = []
        for fold_idx in range(2):
            train_idx = data_args.batch_nodes.get_train_idx()
            test_idx = data_args.batch_nodes.get_test_idx()
            train_node_embeddings = node_embeddings[train_idx]
            
            test_node_embeddings = node_embeddings[test_idx]
            train_labels = data_args.batch_nodes.label[train_idx]
            test_labels = data_args.batch_nodes.label[test_idx]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_node_embeddings, train_labels)
            ACC = cls.score(test_node_embeddings, test_labels)
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
    train_loss = 0.0
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        train_loss = single_epoch_training_util(data_args, model_args, args)
        cost_loss.append(train_loss)
        mean_10folds, std_10folds = evaluate(epoch, data_args, model_args, args)
        print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | mean {:5.2f} | std {:5.2f} | '.format(
                    epoch, (time.time() - epoch_start_time), train_loss, mean_10folds*100, std_10folds*100))
        if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
            model_args.scheduler.step()
        if(mean_10folds>mean_10folds_best):
            
            mean_10folds_best = mean_10folds
            std_10folds_best = std_10folds
    return mean_10folds_best, std_10folds_best