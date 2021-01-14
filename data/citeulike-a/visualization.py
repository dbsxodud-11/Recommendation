import pandas as pd
import matplotlib.pyplot as plt

def visualize() :

    CDL_sparse_accuracy = list(pd.read_csv("CDL_sparse_dense.csv").values[:, 1])
    SVD_sparse_accuracy = list(pd.read_csv("SVD_sparse_dense.csv").values[:, 1])

    M = [50, 100, 150, 200, 250, 300]
    plt.plot(M, CDL_sparse_accuracy, label="CDL", linestyle="dashed", color="black")
    plt.plot(M, SVD_sparse_accuracy, label="SVD", linestyle="dashed", color="green")
    plt.xlabel("M")
    plt.ylabel("recall@M")
    plt.legend()
    plt.savefig("dense.png")

if __name__ == "__main__" :

    visualize()