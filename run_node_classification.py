import torch
import numpy as np
import sys
# script specific requirement in readme
sys.path.insert(0,"/home/ksingh/courses/master_thesis/Graph-Transformer/U2GNN_pytorch/log_uniform")

from U2GNN_pytorch import ml_node_train_utils
from U2GNN_pytorch import util
from U2GNN_pytorch.metrics import print_evaluation_from_embeddings, print_evaluation

log_path = "/home/ksingh/courses/master_thesis/runs/u2gnn/{}"

args={}
args['dataset']="UCI"
args['batch_size']=200
args['multiplex_folder_path'] = "/home/ksingh/courses/master_thesis/multiplex_datasets"
args['num_epochs']=30
args["ng_data"] = "/home/keshav/courses/master_thesis/Graph-Transformer/code_m/data/NGs.mat"
args['num_neighbors']=12
args['loss_type'] = 'contrastive'
args['model_type'] = 'u2gnn'
args['single_layer_only'] = False
args['ml_model_type'] = 'siamese'
args['projection_dim'] = -1
args['train_fraction'] = 0.10
args['size_x'] = 30
args['eval_type'] = 'kmeans'
args['synth_graph_type'] = "NGs"
args['save_input_list'] = True
args["sampled_num"] = 20
args['num_similarity_neighbors'] = 40
args['create_similarity_layer'] = True
args['scale_features'] = True
args['ml_cluster_mat_folder'] = '/home/ksingh/courses/master_thesis/PM/Datasets'
args = util.Namespace(**args)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device {} for pytorch computation".format(device))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
args.update(device=device)



data_args, args = ml_node_train_utils.data_loading_util(args)


def model_train_evaluate(parameterization):
    model_args = ml_node_train_utils.model_creation_util(parameterization,args,  data_args)
    mean_acc, std = ml_node_train_utils.train_evaluate(data_args,model_args,args)
    return mean_acc

def model_train_evaluate_get_embeds(parameterization):
    model_args = ml_node_train_utils.model_creation_util(parameterization,args,  data_args)
    mean_acc, std = ml_node_train_utils.train_evaluate(data_args,model_args,args)
    node_embeds = ml_node_train_utils.get_node_embeddings(data_args, model_args, args)
    return node_embeds

model_input = {"ff_hidden_size" : 1024, "num_timesteps": 20, "dropout":0.2, "sampled_num":50,"num_hidden_layers":2,"learning_rate":0.05}

embeds = model_train_evaluate_get_embeds(model_input).numpy()
L_arithmetic = np.mean(args.laplacian, axis=2)
print_evaluation(data_args.batch_nodes.label.numpy(), L_arithmetic, K = len(np.unique(data_args.batch_nodes.label.numpy())))
L_gm = util.get_gm(args.laplacian, len(args.graph_obj),args.vocab_size)
print_evaluation(data_args.batch_nodes.label.numpy(), L_gm, K = len(np.unique(data_args.batch_nodes.label.numpy())))
print('saving embeddings')

with open(log_path.format("{}_{}_ml_embeds.npy".format(args.dataset,args.model_type)), 'wb') as f:
    np.save(f, embeds)
