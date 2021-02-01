import numpy as np
import pandas as pd
from scipy import sparse

class ALS :

    def __init__(self, path, n_factor, alpha, reg, k) :

        self.path = path
        self.dataset_name = "MovieLens-1M"
        self.f = n_factor
        self.alpha = alpha
        self.reg = reg
        self.k = k

    def run(self) :

        self.load_data()
        self.train()
        self.evaluate()

    def load_data(self) :

        print("Dataset %s" %self.dataset_name)
        self.n_users = 6040
        self.n_items = 3952
        with open(self.path + "ratings.dat") as f :
            self.matrix = np.zeros((self.n_users, self.n_items))       
            lines = f.readlines()
            for line in lines :
                row, col, val = map(int, line.strip().split("::")[:-1])
                self.matrix[row-1,col-1] = val

        self.data_splitter = np.random.binomial(n=1, p=0.8, size=self.matrix.shape)
        
        self.train_matrix = np.multiply(self.matrix, self.data_splitter)
        self.test_matrix = np.multiply(self.matrix, self.data_splitter^1)

        self.train_pref = np.where(self.train_matrix>0, 1, 0)
        self.train_conf = 1 + self.alpha * self.train_matrix

    def train(self) :

        self.X = np.random.normal(scale=1.0/self.f, size=(self.n_users, self.f))
        self.Y = np.random.normal(scale=1.0/self.f, size=(self.n_items, self.f))
        for episode in range(10) :
            self.X = self.train_by_ALS(self.Y, axis=0)
            self.Y = self.train_by_ALS(self.X, axis=1)

    def train_by_ALS(self, W, axis) :
        n = self.matrix.shape[axis]
        m = self.matrix.shape[axis^1]
        #Precompute
        WTW = np.matmul(W.T, W)
        lambda_I = self.reg * np.eye(self.f)

        for i in range(self.matrix.shape[axis]) :
            if axis == 0 :
                C_W = np.diag(self.train_conf[i,:])
                A = WTW + lambda_I + np.matmul(W.T, np.matmul(C_W - np.identity(m), self.train_pref[i,:]))
                y = np.matmul(np.matmul(W.T, C_W), self.train_pref[i,:])
            else :
                C_W = np.diag(self.train_conf[:,i])
                A = WTW + lambda_I + np.matmul(W.T, np.matmul(C_W - np.identity(n), self.train_conf[:,i]))
                y = np.matmul(np.matmul(W.T, C_W), self.train_prf[:,i])
            W[i] = np.linalg.solve(A, y)
        print(W.shape)
        return W

if __name__ == "__main__" :
    als = ALS("./ml-1m/", n_factor=10, alpha=40, reg=100, k=100)
    als.run()
