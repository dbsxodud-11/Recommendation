import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm
from matplotlib import style

class ALS :
    def __init__(self, data_path, n_factor, alpha, reg, seed=0) :
        self.data_path = data_path
        self.f = n_factor
        self.alpha = alpha
        self.reg = reg
        np.random.seed(seed)
        self.episodes = 20

    def run(self) :
        self.load_data()
        self.train()
        performance = self.evaluate()
        return performance

    def load_data(self) :
        print(f"# of Factors: {self.f}")
        print(f"Dataset Loaded...")

        with open(self.data_path + "u.data") as f :
            data = []
            lines = f.readlines()
            for line in lines :
                line = list(map(int, line.strip().split("\t")[:-1]))
                data.append(line)
        users, items, values = zip(*data)
        n_users = max(users)
        n_items = max(items)

        self.matrix = np.zeros((n_users, n_items))
        for user, item, value in zip(users, items, values) :
            self.matrix[user-1, item-1] = value
        
        # Split Train and Test Data
        data_splitter = np.random.binomial(n=1, p=0.8, size=self.matrix.shape)
        self.train_matrix = np.multiply(self.matrix, data_splitter)
        self.test_matrix = np.multiply(self.matrix, data_splitter^1)

        self.train_csr = sparse.csr_matrix(self.train_matrix)
        self.test_csr = sparse.csr_matrix(self.test_matrix)

    def train(self) :
        self.X = np.random.normal(scale=1.0/self.f, size=(self.matrix.shape[0], self.f))
        self.Y = np.random.normal(scale=1.0/self.f, size=(self.matrix.shape[1], self.f))

        matrix_X = self.train_csr
        matrix_Y = self.train_csr.transpose().tocsr()

        for episode in tqdm(range(self.episodes)) :
            self.X = self.train_by_ALS(self.Y, matrix_X)
            self.Y = self.train_by_ALS(self.X, matrix_Y)
            # Loss
            loss = np.sum(np.multiply((1+self.alpha*self.train_matrix), 
                               np.square(np.where(self.train_matrix>0, 1, 0) - np.matmul(self.X, self.Y.T))))
            loss += self.reg*(np.sum(np.square(self.X)))
            loss += self.reg*(np.sum(np.square(self.Y)))
        print(f"Loss : {loss}")
    
    def train_by_ALS(self, Y, matrix) :
        # Precompute
        YTY_lambdaI = np.matmul(Y.T, Y) + self.reg*np.identity(self.f)
        A = np.repeat(np.expand_dims(YTY_lambdaI, axis=0), matrix.shape[0], axis=0)
        Y_lookup = Y[matrix.indices] # User가 rating한 item에 해당하는 y_i만 추출
        R_lookup = matrix.data
        expanded_lookup = np.expand_dims(Y_lookup, axis=1)
        A1 = np.matmul(np.transpose(expanded_lookup, axes=[0, 2, 1]), expanded_lookup) # User가 rating한 item에 해당하는 y_i^T*y_i를 계산
        A2 = np.multiply(R_lookup.reshape(-1, 1, 1), A1) # r_ui * y_i^T * y_i
        y = np.empty((matrix.shape[0], self.f), dtype="float32")
        for idx in range(matrix.shape[0]) :
            beg, end = matrix.indptr[idx], matrix.indptr[idx+1]
            A[idx] += self.alpha * np.sum(A2[beg:end], axis=0)
            y[idx] = np.sum(Y_lookup[beg:end], axis=0) + self.alpha * np.sum(np.multiply(Y_lookup[beg:end], R_lookup[beg:end].reshape(-1, 1)), axis=0)
        return np.linalg.solve(A, y)
    
    def evaluate(self) :
        pred_pref = np.matmul(self.X, self.Y.T)
        ranking = self.matrix.shape[1] - np.argsort(pred_pref) - 1

        percentile_ranking = np.sum(np.multiply(ranking, self.test_matrix)) / np.sum(self.test_matrix)
        percentile_ranking /= self.matrix.shape[1]
        print(f"Percentile Ranking : {percentile_ranking}")
        return percentile_ranking

if __name__ == "__main__" :
    performance_list = []
    n_factors = [10, 20, 40, 80, 160]
    for n_factor in n_factors :
        als = ALS("../_data/ml-100k/", n_factor=n_factor, alpha=40, reg=1000, seed=99)
        performance = als.run()
        performance_list.append(performance)

    style.use("ggplot")
    plt.plot(n_factors, performance_list, linestyle = "dashed", label = "ALS", color = "mediumpurple")
    plt.title("Percentile Ranking of ALS")
    plt.xlabel("# factors")
    plt.ylabel("Expected Percentile Ranking")
    plt.legend()
    plt.savefig("../_plots/ALS(scipy_improved)_performance.png")

