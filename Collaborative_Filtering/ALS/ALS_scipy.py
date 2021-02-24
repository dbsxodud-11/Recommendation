import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        
        self.train_matrix = np.where(np.multiply(self.matrix, self.data_splitter) > 0, 1, 0)
        self.train_csr = sparse.csr_matrix(self.train_matrix)
        self.test_matrix = np.where(np.multiply(self.matrix, self.data_splitter^1) > 0, 1, 0)
        self.test_csr = sparse.csr_matrix(self.test_matrix)

        self.train_pref = np.where(self.train_matrix>0, 1, 0)
        self.train_conf = 1 + self.alpha * self.train_matrix

    def train(self) :

        self.X = np.random.normal(scale=1.0/self.f, size=(self.n_users, self.f))
        self.Y = np.random.normal(scale=1.0/self.f, size=(self.n_items, self.f))
        mat_Y = self.train_csr
        mat_X = self.train_csr.transpose(copy=True).tocsr()
        for episode in tqdm(range(20)) :
            self.X = self.train_by_ALS(self.Y, mat_Y)
            self.Y = self.train_by_ALS(self.X, mat_X)
            # Calculate Loss
            loss = np.sum(np.multiply(self.train_conf, np.square(self.train_pref-np.matmul(self.X, self.Y.T)))) + self.reg * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
            # print("Episode :  %d   Loss : %f" %(episode, loss))
        print(loss)

    def train_by_ALS(self, W, mat) :
        W2 = np.matmul(W.T, W)
        A0 = W2 + np.eye(self.f) * self.reg
        A = np.repeat(np.expand_dims(A0, axis=0), mat.shape[0], axis=0)
        lookup = W[mat.indices]
        expanded_lookup = np.expand_dims(lookup, axis=1)
        A1 = np.matmul(np.transpose(expanded_lookup, axes=[0, 2, 1]), expanded_lookup)
        y = np.empty((mat.shape[0], self.f), dtype="float32")
        for idx in range(mat.shape[0]):
            beg, end = mat.indptr[idx], mat.indptr[idx + 1]
            A[idx] += self.alpha * np.sum(A1[beg: end], axis=0)
            y[idx] = (1.0 + self.alpha) * np.sum(lookup[beg: end], axis=0)
        ret = np.linalg.solve(A, y)
        return ret

    def evaluate(self) :

        pred_pref = np.matmul(self.X, self.Y.T)
        ranking = np.argsort(-pred_pref, axis=1)

        percentile_ranking = np.sum(np.multiply(ranking.argsort(), self.test_matrix)) / np.sum(self.test_matrix)
        percentile_ranking /= self.n_items
        return percentile_ranking


if __name__ == "__main__" :
    performance_list = []
    n_factors = [10, 20, 30, 40, 50]
    for n_factor in n_factors :
        als = ALS("../_data/ml-100k/", n_factor=n_factor, alpha=10, reg=100, seed=10)
        performance = als.run()
        performance_list.append(performance)

    plt.plot(n_factors, performance_list, linestyle = "dashed", label = "ALS", color = "black")
    plt.title("Percentile Ranking of ALS")
    plt.xlabel("# factors")
    plt.ylabel("Expected Percentile Ranking")
    plt.legend()
    plt.savefig("../_plots/ALS(scipy)_performance.png")