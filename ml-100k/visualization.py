import pandas as pd
import matplotlib.pyplot as plt

def visualize() :

    rmse = list(sorted(pd.read_csv("AutoRec_accuracy(hidden).csv").values[:, -1], reverse=True))

    K = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    plt.plot(K, rmse, label="AutoRec", linestyle="dashed", color="green")
    plt.xlabel("number of hidden units")
    plt.ylabel("RMSE")
    plt.savefig("rmse.png")

if __name__ == "__main__" :
    visualize()