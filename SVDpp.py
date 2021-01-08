import numpy as np
import pandas as pd
import math
from utils import *

def SVDpp(train_data, test_data, n_factors) :

    train_df, n_users, n_items = get_data(train_data)

    # New Neighborhood method proposed in https://dl.acm.org/doi/pdf/10.1145/1401890.1401944
    r_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        r_ui[user_id-1][item_id-1] = rating
    r_ui = np.array(r_ui)

    mu = np.mean(list(train_df.rating.values))
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # Initialize Factors 
    q_i = [np.random.normal(0, 0.1, n_factors) for i in range(n_items)]
    p_u = [np.random.normal(0, 0.1, n_factors) for i in range(n_users)]
    n_ui = [[] for _ in range(n_users)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            if r_ui[user_id][item_id] != 0 :
                n_ui[user_id].append(item_id)
    y_j = np.zeros((n_items, n_factors))

    lr_all = 0.005
    lr_b_u = lr_all
    lr_b_i = lr_all
    lr_q_i = lr_all
    lr_p_u = lr_all
    lr_y_j = lr_all

    reg_all = 0.02
    reg_b_u = reg_all
    reg_b_i = reg_all
    reg_q_i = reg_all
    reg_p_u = reg_all
    reg_y_j = reg_all

    max_episode = 100
    performance_list = []

    for episode in range(max_episode) :

        target_value = 0
        for user_id, item_id, rating in list(train_df.values) :

            w_ui = np.zeros(n_factors)
            for item_j in n_ui[user_id-1] :
                w_ui += y_j[item_id-1]
            w_ui /= math.sqrt(len(n_ui[user_id-1]))
            r_hat_ui = mu + b_u[user_id-1] + b_i[item_id-1] + np.dot(q_i[item_id-1], p_u[user_id-1] + w_ui)

            err_ui = rating - r_hat_ui
            target_value += err_ui**2

            # Update
            b_u[user_id-1] += lr_b_u*(err_ui - reg_b_u*b_u[user_id-1])
            b_i[item_id-1] += lr_b_i*(err_ui - reg_b_i*b_i[item_id-1])
            q_i[item_id-1] += lr_q_i*(err_ui*w_ui - reg_q_i*q_i[item_id-1])
            p_u[user_id-1] += lr_p_u*(err_ui*q_i[item_id-1] - reg_p_u*p_u[user_id-1])

            for item_j in n_ui[user_id-1] :
                # print(y_j[item_j])
                # print((err_ui/math.sqrt(len(n_ui[user_id-1])))*q_i[item_id-1] - reg_y_j*y_j[item_j])
                y_j[item_j] += lr_y_j*((err_ui/math.sqrt(len(n_ui[user_id-1])))*q_i[item_id-1] - reg_y_j*y_j[item_j]) 

        print(episode+1, target_value)
    
    # Test
    test_df, n_users, n_items = get_data(test_data)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        w_ui = 0
        for item_j in n_ui[user_id-1] :
            w_ui += y_j[item_j]
        w_ui /= math.sqrt(len(n_ui[user_id-1]))
        rmse += (rating - (mu + b_u[user_id-1] + b_i[item_id-1] + np.dot(q_i[item_id-1], p_u[user_id-1] + w_ui)))**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.4875874731082927


if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    n_factors = 10
    SVDpp(train_data, test_data, n_factors)