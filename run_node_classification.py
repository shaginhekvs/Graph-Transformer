import torch
import numpy as np
import sys
# script specific requirement in readme
sys.path.insert(0,"/home/keshav/courses/master_thesis/Graph-Transformer/U2GNN_pytorch/log_uniform")

from U2GNN_pytorch import node_train_utils
from U2GNN_pytorch import util


args={}
args['dataset']="cora"
args['batch_size']=-1
args['num_epochs']=100
args['num_neighbors']=10

args = util.Namespace(**args)


device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print("using device {} for pytorch computation".format(device))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
args.update(device=device)



data_args, args = node_train_utils.data_loading_util(args)


def model_train_evaluate(parameterization):
    model_args = node_train_utils.model_creation_util(parameterization,args)
    mean_acc, std = node_train_utils.train_evaluate(data_args,model_args,args)
    return mean_acc

model_input = {"ff_hidden_size" : 300, "num_timesteps": 5, "dropout":0.5, "sampled_num":50,"num_hidden_layers":2,"learning_rate":0.0001}

model_train_evaluate(model_input)