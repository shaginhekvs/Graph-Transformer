
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import Namespace
from .contrastive_loss import GraphContrastiveLoss

class Loss_functions(nn.Module):
    """
    Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, loss_func_name, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.loss_func_name = loss_func_name
        if(loss_func_name == 'gae'):
            self.forward_func = gae_loss
        elif(loss_func_name == 'contrastive'):
            self.forward_obj = GraphContrastiveLoss(temperature = temperature)
            self.forward_func = contrastive_loss
            
            
    def forward(self,args):
        if(self.loss_func_name == 'gae'):
            return self.forward_func(args)
        elif(self.loss_func_name == 'contrastive'):
            args.update(contrastiveObj = self.forward_obj)
            return self.forward_func(args)
        else:
            raise NotImplementedError('unknown loss {}'.format(self.loss_func_name))
        
            

def contrastive_loss(args):
    loss_this = torch.tensor(0.0,dtype  = torch.float32).to(args.device)
    for i,logits in enumerate(args.logits_list):
        args_loss = Namespace(features = logits, mask = args.adj_mat[:,:,i])
        loss_this += args.contrastiveObj(args_loss)
    return loss_this
        
def gae_loss(args):
    A_preds = []
    for logits in args.logits_list:
        A_preds.append(torch.sigmoid(torch.matmul(logits,logits.t())))
    A_pred = torch.stack(A_preds,dim=2)
    loss_val = args.norm*F.binary_cross_entropy(A_pred.view(-1), args.adj_label.view(-1), weight = args.weight_tensor)
    
    
    return loss_val




