import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import style

class MLP(nn.Module) :
    def __init__(input_dim, output_dim, hidden_dim, lambda_w) :
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleDict()

        input_dims = [input_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]
        for in_dim, out_dim in zip(input_dims, output_dims) :
            self.layers.append(nn.Linear(in_dim, out_dim))
        self.init_layers()

        self.activations["sigmoid"] = nn.Sigmoid()
        self.activations["dropout"] = nn.Dropout(p=0.1)

    def init_layers(self) :
        for layer in self.layers :
            torch.nn.init.normal_(layer.weight.data, 0.0, 1/lambda_w)
            torch.nn.init.normal_(layer.bias.data, 0.0, 1/lambda_w)

    def encode(self, x) :
        for i, layer in enumerate(self.layers[:2]) :
            x = layer(x)
            x = self.activations["sigmoid"](x)
            x = self.activations["dropout"](x)
        return x

    def decode(self, x) :
        for i, layer in enumerate(self.layers[2:]) :
            x = layer(x)
            x = self.activations["sigmoid"](x)
            x = self.activations["dropout"](x)
        return x

class CDL :
    def __init__(self, rating_matrix, item_info_matrix) :
        super(CDL, self).__init__()

        np.random.seed(42)
        self.input_dm = item_info_matrix.shape[1]
        self.hidden_dim_1 = 200
        self.hidden_dim_2 = 50
        self.k = 50

        self.lambda_w = 0.1
        self.lambda_n = 10
        self.lambda_u = 1
        self.lambda_v = 10

        self.a = 1
        self.b = 0.01
        self.P = 1 # Sparse and Dense Setting
        self.lr = 0.01

        self.n_users = rating_matrix.shape[0]
        self.n_items = rating_matrix.shape[1]

        self.SDAE = MLP(self.input_dim, self.input_dim, hidden_dim=[self.hidden_dim_1, self.hidden_dim_2], lambda_w=self.lambda_w) 

        self.item_info_matrix = item_info_matrix
        self.rating_matrix = rating_matrix

        self.loss_ftn = nn.MSELoss(reduction="sum")

        # Get Training Set
        for i in range(self.n_users) :
            x = np.random.choice(np.where(self.rating_matrix[i,:]>0)[0], self.P)
            self.rating_matrix[i, x] = 1
        
        # Get Confidence Matrix
        self.confidence = np.zeros(self.rating_matrix.shape)
        self.confidence.fill(self.b)
        self.confidence[np.where(self.rating_matrix>0)] = self.a

        # Initialize Latent Vectors
        self.U = torch.normal(0, 1/self.lambda_u, size=(self.n_users, self.k))
        self.V = torch.normla(0, 1/self.lambda_v, size=(self.n_items, self.k))

        self.epochs = 200
        self.batch_size = 120
        self.optimizer = optim.Adam(lself.SDAE.parameters(), lr=self.lr, weight_decay=self.lambda_w)
    
    def train_model(self) :
        self.X_0 = self.add_noise(self.item_info_matrix)
        self.X_0 = torch.tensor(self.X_0, dtype=torch.float32)

        self.X_c = self.item_info_matrix
        self.X_c = torch.tensor(X_c, dtype=torch.float32)

        random_idx = np.random.permutation(self.n_items)
        for epoch in self.epochs :
            cost = 0.0
            for i in range(0, self.n_items, self.batch_size) :
                batch_idx = random_idx[i:i+self.batch_size]

                batch_X_0 = self.X_0[batch_idx, :]
                batch_X_2 = self.SDAE.encode(batch_X_0)
                batch_X_L = self.SDAE.decode(batch_X_2)
                batch_X_c = self.X_c[batch_idx, :]

                batch_R = self.rating_matrix[:, batch_idx]
                batch_R = torch.tensor(batch_R, dtype=torch.float32)
                batch_C = self.confidence[:, batch_idx]
                batch_C = torch.tensor(batch_C, dtype=torch.float32)

                loss = 0.0
                loss_1 = 0.5*self.loss_ftn()