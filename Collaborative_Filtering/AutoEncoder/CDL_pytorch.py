import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class MLP(nn.Module) :

    def __init__(self, input_dims, output_dims, lambda_w) :
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        for input_dim, output_dim in zip(input_dims, output_dims) :
            layer = nn.Linear(input_dim, output_dim)
            layer.weight.data = nn.Parameter(torch.normal(0, 1/lambda_w, size=(output_dim, input_dim)))
            layer.bias.data = nn.Parameter(torch.normal(0, 1/lambda_w, size=(283, output_dim)))
            self.layers.append(layer)

        self.activations = nn.ModuleDict({
            "sigmoid" : nn.Sigmoid(),
            "dropout" : nn.Dropout(p=0.1)
        })

    def encode(self, x) :

        for i, layer in enumerate(self.layers[:2]) :
            # print(x.shape)
            x = layer(x)
            x = self.activations["sigmoid"](x)
            x = self.activations["dropout"](x)

        return x

    def decode(self, x) :

        for i, layer in enumerate(self.layers[2:]) :
            x = layer(x)
            # print(x.shape)
            x = self.activations["sigmoid"](x)
            x = self.activations["dropout"](x)

        return x

class CDL(nn.Module) :

    def __init__(self, rating_matrix, item_info_matrix) :
        super(CDL, self).__init__()

        np.random.seed(42)

        # HyperParameter
        self.n_input = item_info_matrix.shape[1]
        self.n_hidden_1 = 200
        self.n_hidden_2 = 50 # L = 2
        self.k = 50

        self.lambda_w = 0.1
        self.lambda_n = 10
        self.lambda_u = 1
        self.lambda_v = 10

        self.drop_ratio = 0.1
        self.learning_rate = 0.01

        self.a = 1
        self.b = 0.01
        self.P = 10 # sparse(1) and dense(10) setting

        self.n_users = rating_matrix.shape[0]
        self.n_items = rating_matrix.shape[1]

        self.U = np.random.normal(size=(self.n_users, self.k))
        self.V = np.random.normal(size=(self.n_items, self.k))

        # Deep Learning Network
        input_dims = [self.n_input, self.n_hidden_1, self.n_hidden_2, self.n_hidden_1]
        output_dims = [self.n_hidden_1, self.n_hidden_2, self.n_hidden_1, self.n_input]
        self.neural_network = MLP(input_dims, output_dims, self.lambda_w)

        self.item_info_matrix = item_info_matrix
        self.rating_matrix = rating_matrix

        # Get Training Set
        for i in range(self.n_users) :
            x = np.random.choice(np.where(self.rating_matrix[i, :]>0)[0], self.P)
            self.rating_matrix[i, x] = 1

        # Get Confidence Matrix
        self.confidence = self.b * np.ones((self.n_users, self.n_items))
        self.confidence[np.where(self.rating_matrix>0)] = self.a

        self.epochs = 200
        self.batch_size = 283

        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)

    def train_model(self) :

        self.X_0 = self.add_noise(self.item_info_matrix)
        self.X_0 = torch.tensor(self.X_0, dtype=torch.float32)

        self.X_c = self.item_info_matrix
        self.X_c = torch.tensor(self.X_c, dtype=torch.float32)

        random_idx = np.random.permutation(self.n_items)

        for epoch in tqdm(range(self.epochs)) :
            batch_cost = 0

            for i in range(self.n_users) :
                c_diag = np.diag(self.confidence[i, :])
                term_1 = np.matmul(np.matmul(self.V.T, c_diag), self.V) + self.lambda_u*np.identity(self.k)
                term_2 = np.matmul(np.matmul(self.V.T, c_diag), self.rating_matrix[i, :])
                self.U[i, :] = np.linalg.solve(term_1, term_2)#np.matmul(np.linalg.inv(term_1), term_2)

            for j in range(self.n_items) :
                c_diag = np.diag(np.confidence[:, j])
                term_1 = np.matmul(np.matmul(self.U.T, c_diag), self.U) + self.lambda_v*np.identity(self.k)
                term_2 = np.matmul(np.matmul(self.U.T, c_diag), self.rating_matrix[:, j].reshape(-1, 1))
                self.V[:, j] = np.mamtul(np.linalg.inv(term_1), term_2)

            for i in range(0, self.n_items, self.batch_size) :
                batch_idx = random_idx[i:i+self.batch_size]
                
                batch_X_0 = self.X_0[batch_idx, :]
                batch_X_2 = self.neural_network.encode(batch_X_0)
                batch_X_4 = self.neural_network.decode(batch_X_2)
                batch_X_c = self.X_c[batch_idx, :]

                batch_R = self.rating_matrix[:, batch_idx]
                batch_R = torch.tensor(batch_R, dtype=torch.float32)
                batch_C = self.confidence[:, batch_idx]
                batch_C = torch.tensor(batch_C, dtype=torch.float32)

                batch_V = self.V[batch_idx, :]

                l2_loss = nn.MSELoss(reduction="sum")
                
                loss_1 = 0.5*self.lambda_u*l2_loss(self.U, torch.zeros((self.n_users, self.k)))
        
                loss_2 = 0.0
                for j, layer in enumerate(self.neural_network.layers) :
                    loss_2 += l2_loss(layer.weight, torch.zeros(layer.weight.data.shape)) + l2_loss(layer.bias, torch.zeros(layer.bias.data.shape))
                loss_2 = 0.5*self.lambda_w*loss_2

                loss_3 = 0.5*self.lambda_v*l2_loss(batch_V, batch_X_2)

                loss_4 = 0.5*self.lambda_n*l2_loss(batch_X_c, batch_X_4)

                batch_err = batch_R - torch.matmul(self.U, torch.transpose(batch_V, 0, 1))
                loss_5 = 0.5*torch.sum(torch.mul(batch_C, torch.mul(batch_err, batch_err)))

                loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_cost += loss.item()


            print(epoch+1, batch_cost)

        return torch.matmul(self.U, torch.transpose(self.V, 0, 1)).detach().numpy()

    def add_noise(self, item_info_matrix) :

        x = item_info_matrix
        noise = np.random.binomial(1, 0.7, size = (x.shape[0], x.shape[1]))
        x *= noise

        return x

def get_performance(true_matrix, pred_matrix) :

    # reacall@ performance
    M = [50, 100, 150, 200, 250, 300]
    accuracy = []
    for m in M :
        all_cnt = 0
        for i in range(true_matrix.shape[0]):
            l_score = np.ravel(pred_matrix[i, :]).tolist()
            pl = sorted(enumerate(l_score), key=lambda d : d[1], reverse=True)
            l_rec = [i[0] for i in pl][:m]
            s_rec = set(l_rec)
            s_true = set(np.ravel(np.where(true_matrix[i, :]>0)))
            cnt_hit = len(s_rec.intersection(s_true))
            all_cnt = all_cnt + cnt_hit / len(s_true)
        accuracy.append(all_cnt / true_matrix.shape[0])

    plt.plot(M, accuracy, linestyle = "dashed", label = "CDL", color = "black")
    plt.xlabel("M")
    plt.ylabel("Recall")
    plt.title("Performance of CDL in sparse setting")
    plt.legend()
    plt.savefig("./_data/citeulike-a/CDL_sparse.png")

    df = pd.DataFrame(accuracy, columns=["accuracy"])
    df.to_csv("./_data/citeulike-a/CDL_sparse_accuracy.csv")                
                

if __name__ == "__main__" :

    # Data Processing
    with open("../_data/citeulike-a/vocabulary.dat", "r") as vocabulary_file :
        n_vocabulary = len(vocabulary_file.readlines())

    with open("../_data/citeulike-a/mult.dat", "r") as item_info_file :
        n_items = len(item_info_file.readlines())
    
    item_info_matrix = np.zeros((n_items, n_vocabulary))
    with open("../_data/citeulike-a/mult.dat", "r") as item_info_file :
        sentences = item_info_file.readlines()       
        for i, sentence in enumerate(sentences) : 
            words = sentence.strip().split(" ")[1:]
            for word in words :
                j, k =word.split(":")
                item_info_matrix[i][int(j)] = int(k)


    with open("../_data/citeulike-a/users.dat", "r") as user_info_file :
        n_users = len(user_info_file.readlines())

    rating_matrix = np.zeros((n_users, n_items))
    with open("../_data/citeulike-a/users.dat", "r") as user_info_file :
        ratings = user_info_file.readlines()
        for i, rating in enumerate(ratings) :
            items = list(map(int, rating.strip().split(" ")))
            for item in items :
                rating_matrix[i][item] = 1

    # Train Model
    # print(rating_matrix[:20, :20])
    rating_matrix_true = rating_matrix.copy()
    cdl = CDL(rating_matrix_true, item_info_matrix)
    rating_matrix_pred = cdl.train_model()
    # print(rating_matrix_pred[:20,:20])
    # print(rating_matrix_true[:20,:20])

    # Get Performance
    get_performance(rating_matrix, rating_matrix_pred)