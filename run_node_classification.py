import torch
import numpy as np
import sys
# script specific requirement in readme
sys.path.insert(0,"/home/ksingh/courses/master_thesis/Graph-Transformer/U2GNN_pytorch/log_uniform")

from U2GNN_pytorch import ml_node_train_utils
from U2GNN_pytorch import util

log_path = "~/courses/master_thesis/runs/u2gnn/{}"

args={}
args['dataset']="cora"
args['batch_size']=-1
args['num_epochs']=2000
args['num_neighbors']=10
args['loss_type'] = 'contrastive'
args['model_type'] = 'u2gnn'
args['single_layer_only'] = False
args['ml_model_type'] = 'siamese'

args = util.Namespace(**args)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device {} for pytorch computation".format(device))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
args.update(device=device)

args = util.Namespace(**args)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device {} for pytorch computation".format(device))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
args.update(device=device)


data_args, args = node_train_utils.data_loading_util(args)


def model_train_evaluate(parameterization):
    model_args = node_train_utils.model_creation_util(parameterization,args)
    mean_acc, std = node_train_utils.train_evaluate(data_args,model_args,args)
    return mean_acc

def model_train_evaluate_get_embeds(parameterization):
    model_args = ml_node_train_utils.model_creation_util(parameterization,args)
    mean_acc, std = ml_node_train_utils.train_evaluate(data_args,model_args,args)
    node_embeds = ml_node_train_utils.get_node_embeddings(data_args, model_args, args)
    return node_embeds

model_input = {"ff_hidden_size" : 1024, "num_timesteps": 5, "dropout":0.5, "sampled_num":50,"num_hidden_layers":4,"learning_rate":0.001}

embeds = model_train_evaluate_get_embeds(model_input).numpy()

print('saving embeddings')

with open(log_path.format("{}_{}_ml_embeds.npy".format(args.dataset,args.model_type)), 'wb') as f:
    np.save(f, embeds)