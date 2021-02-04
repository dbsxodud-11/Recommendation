import numpy as np
import pandas as pd
import math
from utils import *
from copy import deepcopy
import codecs


def SVD(data_dir) :

    # train_data, test_data = data_processing(data_dir)
    data_dir = "./ml-100k/"
    with open(data_dir + "u.info", "r") as info_file :
        lines = info_file.readlines()
        n_users = int(lines[0].strip().split(" ")[0])
        n_items = int(lines[1].strip().split(" ")[0])
    
    R = [[0 for _ in range(n_items)] for _ in range(n_users)]
    # Train Data
    train_R = deepcopy(R)
    with open(data_dir + "u1.base", "r") as train_data_file :
        lines = train_data_file.readlines()
        for line in lines :
            user, item, value = list(map(int, line.strip().split("\t")[:-1]))
            train_R[user-1][item-1] = 1
    train_R = np.array(train_R)

    test_R = deepcopy(R)
    with open(data_dir + "u1.test", "r") as test_data_file :
        lines = test_data_file.readlines()
        for line in lines :
            user, item, value = list(map(int, line.strip().split("\t")[:-1]))
            test_R[user-1][item-1] = 1
    test_R = np.array(test_R)

    # Initialize Factors
    n_factor = 10
    U = [np.random.normal(0, 0.01, n_factor) for i in range(n_users)]
    U = np.array(U)
    V = [np.random.normal(0, 0.01, n_factor) for i in range(n_items)]
    V = np.array(V)

    # Train Model
    max_episode = 20
    lr = 0.01
    reg = 0.02

    for episode in range(max_episode) :
        loss = 0
        for i in range(n_users) :
            for j in range(n_items) :
                if train_R[i, j] != 0 :
                    
                    err_ij = train_R[i, j] - np.dot(U[i], V[j])
                    loss += err_ij**2
                    
                    # Gradient Descent
                    U[i] += lr*(err_ij*V[j] - reg*U[i])
                    V[j] += lr*(err_ij*U[i] - reg*V[j])
        print(episode+1, loss)

    # Evaluate Model
    pred_R = np.matmul(U, V.T)
    print(pred_R.shape)

    rmse = 0
    size = 0
    for i in range(n_users) :
        for j in range(n_items) :
            if test_R[i, j] != 0 :
                rmse += (test_R[i, j] - pred_R[i, j])**2
                size += 1
    rmse = math.sqrt(rmse / size)
    print(rmse)

if __name__ == "__main__" :
    data_dir = "./ml-100k/"
    SVD(data_dir)