import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

class ALS :

    def __init__(self, path, n_factor, alpha, reg, seed) :

        self.path = path
        self.dataset_name = "MovieLens-100K"
        self.f = n_factor
        self.alpha = alpha
        self.reg = reg
        self.seed = seed

    def run(self) :

        self.load_data()
        self.train()
        performance = self.evaluate()
        return performance

    def load_data(self) :

        print("Dataset %s" %self.dataset_name)
        print("# Factor %d" %self.f)
        np.random.seed(self.seed)
        self.n_users = 943
        self.n_items = 1682
        with open(self.path + "u.data") as f :
            self.matrix = np.zeros((self.n_users, self.n_items))       
            lines = f.readlines()
            for line in lines :
                row, col, val = map(int, line.strip().split("\t")[:-1])
                self.matrix[row-1,col-1] = val

        self.data_splitter = np.random.binomial(n=1, p=0.8, size=self.matrix.shape)
        
        self.train_matrix = np.multiply(self.matrix, self.data_splitter)
        self.test_matrix = np.where(np.multiply(self.matrix, self.data_splitter^1) >= 5, 5, 0)


        self.train_pref = np.where(self.train_matrix>0, 1, 0)
        self.train_conf = 1 + self.alpha * self.train_matrix

    def train(self) :

        self.X = np.random.normal(scale=1.0/self.f, size=(self.n_users, self.f))
        self.Y = np.random.normal(scale=1.0/self.f, size=(self.n_items, self.f))
        for episode in range(10) :
            self.train_by_ALS(self.Y, axis=0)
            self.train_by_ALS(self.X, axis=1)
            # Calculate Loss
            loss = np.sum(np.multiply(self.train_conf, np.square(self.train_pref-np.matmul(self.X, self.Y.T)))) + self.reg * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
            print("Episode :  %d   Loss : %f" %(episode, loss))

    def train_by_ALS(self, W, axis) :
        n = self.matrix.shape[axis]
        m = self.matrix.shape[axis^1]
        #Precompute
        WTW = np.matmul(W.T, W)
        lambda_I = self.reg * np.eye(self.f)

        for i in range(self.matrix.shape[axis]) :
            if axis == 0 :
                C_W = np.diag(self.train_conf[i,:])
                A = WTW + lambda_I + np.matmul(W.T, np.matmul(C_W - np.identity(m), W))
                y = np.matmul(np.matmul(W.T, C_W), self.train_pref[i,:])
                self.X[i] = np.linalg.solve(A, y)
            else :
                C_W = np.diag(self.train_conf[:,i])
                A = WTW + lambda_I + np.matmul(W.T, np.matmul(C_W - np.identity(m), W))
                y = np.matmul(np.matmul(W.T, C_W), self.train_pref[:,i])
                self.Y[i] = np.linalg.solve(A, y)

    def evaluate(self) :

        pred_pref = np.matmul(self.X, self.Y.T)
        ranking = np.argsort(-pred_pref, axis=1)

        percentile_ranking = np.sum(np.multiply(ranking.argsort(), self.test_matrix)) / np.sum(self.test_matrix)
        percentile_ranking /= self.n_items
        return percentile_ranking


if __name__ == "__main__" :
    performance_list = []
    n_factors = [10, 20, 40, 80, 160]
    for n_factor in n_factors :
        als = ALS("./ml-100k/", n_factor=n_factor, alpha=40, reg=1.0/n_factor, seed=10)
        performance = als.run()
        performance_list.append(performance)

    plt.plot(n_factors, performance_list, linestyle = "dashed", label = "ALS", color = "black")
    plt.title("Percentile Ranking of ALS")
    plt.xlabel("# factors")
    plt.ylabel("Expected Percentile Ranking")
    plt.legend()
    plt.savefig("./data/ALS_performance.png")