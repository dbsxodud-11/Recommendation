import numpy as np
from scipy import sparse
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class BPR :

    def __init__(self, path, n_factor, lr, reg_u, reg_i, batch_size, seed=30) :

        self.path = path
        self.dataset_name = "MovieLens-100K"
        self.f = n_factor
        self.lr = lr
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.batch_size = 1
        self.seed = seed

    def run(self) :

        self.load_data()
        self.train()
        performance = self.evaluate()
        return performance

    def load_data(self) :

        print(f"Dataset {self.dataset_name}")
        np.random.seed(self.seed)
        print(f"# Factors : {self.f}")
        with open(self.path + "u.user", "r") as user_info_file :
            self.n_users = len(user_info_file.readlines())
        with open(self.path + "u.item", "r", encoding="ISO-8859-1") as item_info_file :
            self.n_items = len(item_info_file.readlines())
        with open(self.path + "u.data", "r") as f :
            self.matrix = np.zeros((self.n_users, self.n_items))
            lines = f.readlines()
            for line in lines :
                row, col, val = map(int, line.strip().split("\t")[:-1])
                self.matrix[row-1, col-1] = val
        # print(self.matrix[:10,:10])
        self.data_splitter = np.random.binomial(n=1, p=0.8, size=self.matrix.shape)
        self.train_matrix = np.where(np.multiply(self.matrix, self.data_splitter) > 0, 1, 0)
        # print(train_matrix[:10,:10])
        self.train_matrix = sparse.csr_matrix(self.train_matrix)
        self.test_matrix = np.where(np.multiply(self.matrix, self.data_splitter^1) > 0, 1, 0)
        # print(test_matrix[:10,:10])
        # self.test_matrix = sparse.csr_matrix(test_matrix)

    def train(self) :

        self.U = np.random.normal(scale=1.0/self.f, size=(self.n_users, self.f))
        self.I = np.random.normal(scale=1.0/self.f, size=(self.n_items, self.f))
        
        # batch_iter = len(self.train_matrix.indices) // self.batch_size
        for episode in tqdm(range(500)) :
            for _ in range(self.n_users):
                batch_uij = self.sample()
                self.sgd(batch_uij)

    def sample(self) :

        indptr = self.train_matrix.indptr
        indices = self.train_matrix.indices

        users = np.random.choice(self.n_users, size=self.batch_size)
        pos_items = []
        neg_items = []
        for idx, user in enumerate(users) :
            p_items = indices[indptr[user]: indptr[user+1]]
            p_item = np.random.choice(p_items)
            n_item = np.random.choice(self.n_items)
            while n_item in p_items :
                n_item = np.random.choice(self.n_items)
            pos_items.append(p_item)
            neg_items.append(n_item)
        pos_items = np.array(pos_items)
        neg_items = np.array(neg_items)
        return users, pos_items, neg_items

    def sgd(self, batch_uij) :

        batch_u, batch_i, batch_j = batch_uij
        batch_U = self.U[batch_u]
        batch_I= self.I[batch_i]
        batch_J = self.I[batch_j]

        x_uij = np.sum(np.multiply(batch_U, batch_I-batch_J), axis=1)
        X = np.exp(-x_uij) / (1.0 + np.exp(-x_uij))
        expand_X = np.expand_dims(X, 1)

        grad_U = expand_X*(batch_J-batch_I) + self.reg_u*batch_U
        grad_I = expand_X*(-batch_U) + self.reg_i*batch_I
        grad_J = expand_X*(batch_U) + self.reg_i*batch_J

        self.U[batch_u] -= self.lr*grad_U
        self.I[batch_i] -= self.lr*grad_I
        self.I[batch_j] -= self.lr*grad_J

    def evaluate(self) :

        indptr = self.train_matrix.indptr
        indices = self.train_matrix.indices

        auc = 0.0
        self.train_matrix = self.train_matrix.toarray()
        for i in range(self.n_users) :
            y_pred = np.matmul(self.U[i], self.I.T)
            y_true = self.train_matrix[i]
            if np.sum(y_true) <= 0.0 :
                continue
            auc += roc_auc_score(y_true, y_pred)

        auc /= self.n_users
        print(f"AUC : {auc}")
        return auc

if __name__ == "__main__" :

    n_factors = [10, 20, 50, 100]
    performance_list = []
    for n_factor in n_factors :
        bpr = BPR("./ml-100k/", n_factor=n_factor, lr=0.1, reg_u=0.02, reg_i=0.02, batch_size=120)
        performance = bpr.run()
        performance_list.append(performance)

    plt.plot(performance_list, label="BPR-MF", color = "black", marker="s", linestyle="dashed")
    plt.xticks(np.arange(4), n_factors)
    plt.legend()
    plt.savefig("./data/BPR_MF(AUC curve)")