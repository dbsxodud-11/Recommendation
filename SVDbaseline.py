import numpy as np
import pandas as pd
import math
from utils import *
from copy import deepcopy

def SVDbaseline(train_data, test_data, n_factors) :

    # train_df, n_users, n_items = get_data(train_data)

    # # Latent Factor Model induced by Single Value Decomposition    
    # r_ij = [[0 for _ in range(n_items)] for _ in range(n_users)]
    # for i, j, rating in list(train_df.values) :
    #     r_ij[i-1][j-1] = rating
    # r_ij = np.array(r_ij)
    with open("./data/citeulike-a/mult.dat", "r") as item_info_file :
        n_items = len(item_info_file.readlines())

    with open("./data/citeulike-a/users.dat", "r") as user_info_file :
        n_users = len(user_info_file.readlines())

    r_ui = np.zeros((n_users, n_items))
    with open("./data/citeulike-a/users.dat", "r") as user_info_file :
        ratings = user_info_file.readlines()
        for i, rating in enumerate(ratings) :
            items = list(map(int, rating.strip().split(" ")))
            for item in items :
                r_ui[i][item] = 1

    r_ui_true = deepcopy(r_ui)

    for i in range(n_users) :
            x = np.random.choice(np.where(r_ui[i, :]>0)[0], 10)
            r_ui[i, x] = 1

    # Initialize Biases
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # Initialize Factors
    u_bar = np.array(r_ui.mean(axis = 1)).reshape(-1, 1)
    i_bar = np.array(r_ui.mean(axis = 0)).reshape(-1, 1)
    
    q_u = [np.random.normal(0, 0.1, n_factors) for i in range(n_users)]
    q_u = np.array(q_u)
    p_i = [np.random.normal(0, 0.1, n_factors) for i in range(n_items)]
    p_i = np.array(p_i)

    mu = np.mean(r_ui)

    # Use Gradient Descent Method
    max_episode = 20

    lr = 0.005
    lambda_3 = 0.02
    performance_list = []

    for episode in range(max_episode) :

        target_value = 0
        for i in range(n_users) :
            for j in range(n_items) :
                if r_ui[i][j] != 0 :

                    r_hat_ui = mu + b_u[i] + b_i[j] + np.dot(q_u[i], p_i[j])

                    err_ui = r_ui[i][j] - r_hat_ui
                    target_value += (err_ui)**2

                    # Update
                    b_u[i] += lr*(err_ui - lambda_3*b_u[i])
                    b_i[j] += lr*(err_ui - lambda_3*b_i[j])
                    q_u[i] += lr*(err_ui*p_i[j] - lambda_3*q_u[i])
                    p_i[j] += lr*(err_ui*q_u[i] - lambda_3*p_i[j])

        print(episode+1, target_value)

    # # Test
    # test_df, n_users, n_items = get_data(test_data)

    # # RMSE : test metric
    # rmse = 0
    # size = len(test_df.values)
    # for i, j, rating in list(test_df.values) :
    #     rmse += (rating - (mu + b_u[i-1] + b_i[j-1] + np.dot(q_u[i-1], p_i[j-1])))**2
    # rmse = math.sqrt(rmse / size)
    # print(rmse) # 0.4875874731082927

    pred_matrix = np.zeros((n_users, n_items))
    for i in range(n_users) :
        for j in range(n_items) :
            pred_matrix[i][j] = mu + b_u[i] + b_i[j] + np.dot(q_u[i], p_i[j])

    M = [50, 100, 150, 200, 250, 300]
    accuracy = []
    for m in M :
        all_cnt = 0
        for i in range(r_ui_true.shape[0]):
            l_score = np.ravel(pred_matrix[i, :]).tolist()
            pl = sorted(enumerate(l_score), key=lambda d : d[1], reverse=True)
            l_rec = [i[0] for i in pl][:m]
            s_rec = set(l_rec)
            s_true = set(np.ravel(np.where(r_ui_true[i, :]>0)))
            cnt_hit = len(s_rec.intersection(s_true))
            all_cnt = all_cnt + cnt_hit / len(s_true)
        accuracy.append(all_cnt / r_ui_true.shape[0])

    # plt.plot(M, accuracy, linestyle = "dashed", label = "CDL", color = "black")
    # plt.xlabel("M")
    # plt.ylabel("Recall")
    # plt.title("Performance of CDL in sparse setting")
    # plt.legend()
    # plt.savefig("./data/citeulike-a/CDL_sparse.png")

    df = pd.DataFrame(accuracy, columns=["accuracy"])
    df.to_csv("./data/citeulike-a/SVD_dense_accuracy.csv")  

if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    n_factors = 100
    SVDbaseline(train_data, test_data, n_factors)
