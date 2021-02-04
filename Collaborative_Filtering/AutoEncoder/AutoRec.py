import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import math

class MLP(nn.Module) :

    def __init__(self, input_dim, output_dim, hidden_dim, layer_type, activ_type) :
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        input_dims = [input_dim] + [hidden_dim]
        output_dims = [hidden_dim] + [output_dim]

        for in_dim, out_dim in zip(input_dims, output_dims) :
            if layer_type == "linear" :
                self.layers.append(nn.Linear(in_dim, out_dim))
            else :
                self.layers.append(nn.Identity(in_dim, out_dim))

        if activ_type == "relu" :
            self.activation = nn.ReLU()
        elif activ_type == "sigmoid" :
            self.activation = nn.Sigmoid()

    def forward(self, x) :

        for layer in self.layers :
            x = layer(x)
            x = self.activation(x)

        return x

class AutoRec(nn.Module) :

    def __init__(self, rating_matrix, hidden, layer_type, activ_type) :
        super(AutoRec, self).__init__()

        self.rating_matrix = rating_matrix
        self.rating_mask = np.zeros((self.rating_matrix.shape))
        self.rating_mask[np.where(self.rating_matrix>0)] = 1
        self.input_dim = self.rating_matrix.shape[0]
        self.hidden = hidden
        self.layer_type = layer_type
        self.activ_type = activ_type

        self.autoencoder = MLP(self.input_dim, self.input_dim, self.hidden, self.layer_type, self.activ_type)

        self.epochs = 200
        self.batch_size = 256
        self.lr = 0.01
    
        self.loss_ftn = nn.MSELoss(reduction="sum")
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr = self.lr)

        self.n_users = self.rating_matrix.shape[0]
        self.n_items = self.rating_matrix.shape[1]

    def train_model(self) :

        random_idx = np.random.permutation(self.n_items)
        lambda_1 = 100

        for epoch in range(self.epochs) :
            batch_cost = 0

            for i in range(0, self.n_items, self.batch_size) :
                batch_idx = random_idx[i:i+self.batch_size]
                
                batch_rating_matrix = self.rating_matrix[:, batch_idx].T
                batch_rating_matrix = torch.tensor(batch_rating_matrix, dtype=torch.float32)               
                batch_pred_matrix = self.autoencoder(batch_rating_matrix)
                batch_rating_mask = self.rating_mask[:, batch_idx].T
                batch_rating_mask = torch.tensor(batch_rating_mask, dtype=torch.float32)

                loss_1 = self.loss_ftn(batch_rating_matrix, torch.mul(batch_pred_matrix, batch_rating_mask))
                loss_2 = 0.0
                for layer in self.autoencoder.layers :
                    loss_2 += self.loss_ftn(layer.weight, torch.zeros((layer.weight.shape)))# + self.loss_ftn(layer.bias, torch.zeros((layer.bias.shape)))
                loss_2 = 0.5*lambda_1*loss_2
                # print(loss_1)
                # print(loss_2)
                loss = loss_1 + loss_2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_cost += loss.item()
            print(epoch+1, batch_cost)
            # print(batch_pred_matrix[:4][:4])
        
        # return torch.transpose(self.autoencoder(self.rating_matrix.T), 0, 1)

    def get_performance(self, test_df) :

        pred_matrix = torch.transpose(self.autoencoder(torch.tensor(self.rating_matrix.T, dtype=torch.float32)),0,1)

        # RMSE : test metric
        rmse = 0
        size = len(test_df.values)
        for user_id, item_id, rating in list(test_df.values) :
            rmse += (rating - pred_matrix[user_id-1][item_id-1])**2
        rmse = math.sqrt(rmse / size)
        # print(rmse) # 0.40821710233374997
        return rmse

if __name__ == "__main__" :

    train_data = "./ml-1m/ratings.dat"
    test_data = "./ml-1m/ratings.dat"

    train_df, n_users, n_items = get_data(train_data)
    # print(train_df)
    rating_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]

    for user_id, item_id, rating in list(train_df.values) :
        rating_matrix[user_id-1][item_id-1] = rating
    rating_matrix = np.array(rating_matrix)

    hiddens = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    activ_types = ["relu"]
    accuracy = []
    for hidden in hiddens :
        autorec = AutoRec(rating_matrix, hidden=hidden, layer_type="linear", activ_type="relu")
        autorec.train_model()
        
        test_df, n_users, n_items = get_data(test_data)
        
        accuracy.append(autorec.get_performance(test_df))
    df = pd.DataFrame(accuracy, columns=["accuracy"])
    df.to_csv("./ml-100k/AutoRec_accuracy(hidden).csv")    