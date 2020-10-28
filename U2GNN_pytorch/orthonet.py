import math, random
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def build_net(input_dim, hidden_dim, output_dim):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim//2),
        nn.ReLU(),
        nn.Linear(hidden_dim//2, hidden_dim//3),
        nn.ReLU(),
        nn.Linear(hidden_dim//3, hidden_dim//4),
        nn.ReLU(),
        nn.Linear(hidden_dim//4, output_dim),
    )
    
    return net

class OrthoNet:
    
    def __init__(self, model, epochs, lr):
        self.model  = model
        self.epochs = epochs
        self.lr     = lr
        
    def fit(self, X, L):
        self.setup(L)
        self.train(X)
        
    def setup(self, L):
        self.L = torch.from_numpy(L.astype(np.float32))
        self.m = math.sqrt(len(L))
        self.K = self.model[-1].out_features
        
    def loss(self, inputs):
        z = self.model(inputs)
        M = z.t().mm(z)
        R = torch.cholesky(M, upper=True)
        self.Q = self.m * torch.inverse(R)
        y = torch.mm(z, self.Q)
        loss = y.t().mm(self.L).mm(y).trace()
        return loss
    
    def train(self, X):
        inputs    = X#torch.from_numpy(X.astype(np.float32))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)
        self.history = []
        self.model.train()
        for epoch in range(self.epochs+1):
            loss = self.loss(inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.history.append(loss.item())
            if epoch == 0 or epoch % 500 == 0:
                print('Epoch {:4d}/{:d}: {:2.2f}'.format(epoch, self.epochs, loss.item()))  
        self.model.eval()
    
    def predict(self, X, orthogonal=True):
        inputs  = X
        outputs = self.model(inputs)
        if orthogonal:
            outputs = torch.mm(outputs, self.Q)
        return outputs.data.numpy()