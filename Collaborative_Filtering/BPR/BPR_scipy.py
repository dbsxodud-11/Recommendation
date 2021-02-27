import numpy as np
from scipy import sparse
import logging
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class BPR :
    def __init__(self, data_path, n_factor, lr, reg_u, reg_i, reg_j, seed=0) :
        self.data_path = data_path
        self.f = n_factor
        self.lr = lr
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.reg_j = reg_j
        np.random.seed(seed)
        self.episodes = 50
        self.get_logger()
        

    def get_logger(self) :
        self.logger = logging.getLogger("BPR")
        self.logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def run(self) :
        self.load_data()
        self.train()
        performance = self.evaluate()
        return performance

    def load_data(self):
        start_time = time.time()
        with open(self.data_path + "u.data") as f :
            data = []
            lines = f.readlines()
            for line in lines :
                line = list(map(int, line.strip().split("\t")[:-1]))
                data.append(line)
        users, items, values = zip(*data)
        n_users = max(users)
        n_items = max(items)
        finish_time = time.time()
        self.logger.info(f"DataSet Loaded: {finish_time-start_time:.3f}sec")
        self.logger.info(f"# of Factors: {self.f}")

        start_time = time.time()
        self.matrix = np.zeros((n_users, n_items))
        for user, item, value in zip(users, items, values) :
            self.matrix[user-1, item-1] = value
        self.matrix = np.where(self.matrix >= 4, 1, 0)
        
        # Split Train and Test Data
        data_splitter = np.random.binomial(n=1, p=0.8, size=self.matrix.shape)
        self.train_matrix = np.multiply(self.matrix, data_splitter)
        self.test_matrix = np.multiply(self.matrix, data_splitter^1)

        self.train_csr = sparse.csr_matrix(self.train_matrix)
        self.test_csr = sparse.csr_matrix(self.test_matrix)        
        finish_time = time.time()
        self.logger.info(f"Split Train and Test Tataset: {finish_time-start_time:.3f}sec")

    def train(self) :
        self.U = np.random.normal(scale=1.0/self.f, size=(self.matrix.shape[0], self.f))
        self.I = np.random.normal(scale=1.0/self.f, size=(self.matrix.shape[1], self.f))

        start_time = time.time()
        for episode in tqdm(range(self.episodes)) :
            for u in range(self.matrix.shape[0]) :
                beg, end = self.train_csr.indptr[u], self.train_csr.indptr[u+1]
                pos_items = self.train_csr.indices[beg:end]
                neg_items = np.zeros(self.matrix.shape[1])
                neg_items[pos_items] = 1
                neg_items = np.random.choice(np.where(neg_items == 0)[0], len(pos_items), replace=False)
            
                for i, j in zip(pos_items, neg_items) :
                    x_ui = np.dot(self.U[u], self.I[i])
                    x_uj = np.dot(self.U[u], self.I[j])
                    x_uij = x_ui - x_uj
                    x_uij = np.divide(np.exp(-x_uij), (1 + np.exp(-x_uij)))

                    self.U[u] += self.lr * (x_uij*(self.I[i]-self.I[j]) + self.reg_u*self.U[u])
                    self.I[i] += self.lr * (x_uij*self.U[u] + self.reg_i*self.I[i])
                    self.I[j] += self.lr * (x_uij*(-self.U[u]) + self.reg_j*self.I[j])
        finish_time = time.time()
        self.logger.info(f"Training Completed: {finish_time-start_time:.3f}sec")

    def evaluate(self) :
        auc = 0.0
        for i in range(self.n_users) :
            y_pred = np.matmul(self.U[i], self.I.T)
            y_true = self.matrix[i]
            if np.sum(y_true) < 0.0 :
                continue
            auc += roc_auc_score(y_true, y_pred)
        auc /= self.matrix.shape[0]
        self.logger.info(f"Test AUC: {auc:.3f}")
        return auc

if __name__ == "__main__" :
    performance_list = []
    n_factors = [10]
    for n_factor in n_factors :
        bpr = BPR("../_data/ml-100k/", n_factor=n_factor, lr=0.01, reg_u=0.02, reg_i=0.02, reg_j=0.02, seed=99)
        performance = bpr.run()
        performance_list.append(performance)

    
