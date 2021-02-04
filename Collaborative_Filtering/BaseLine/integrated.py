import numpy as np
import pandas as pd
import math
from utils import *

def integrated(train_data, test_data, n_factors, neighbor_size) :

    train_df, n_users, n_items = get_data(train_data)

    # New Neighborhood method proposed in https://dl.acm.org/doi/pdf/10.1145/1401890.1401944
    r_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    n_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        r_ui[user_id-1][item_id-1] = rating
        n_ui[user_id-1][item_id-1] = 1
    r_ui = np.array(r_ui)
    n_ui = np.array(n_ui)

    mu = np.mean(list(train_df.rating.values))
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # Initialize Factors
    q_i = [np.random.normal(0, 0.1, n_factors) for i in range(n_items)]
    p_u = [np.random.normal(0, 0.1, n_factors) for i in range(n_users)]
    n_u = [[] for _ in range(n_users)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            if r_ui[user_id][item_id] != 0 :
                n_u[user_id].append(item_id)
    y_j = np.zeros((n_items, n_factors))

    w_ij = np.zeros((n_items, n_items))
    c_ij = np.zeros((n_items, n_items))

    # Caluclate Similarity
    lambda_2 = 0.4
    p_ij = np.zeros((n_items, n_items))
    for item_i in range(n_items) :
        for item_j in range(n_items) :
            user_rated_ij = []
            for user_id in range(n_users) :
                if r_ui[user_id][item_i] != 0 and r_ui[user_id][item_j] != 0:
                    user_rated_ij.append(user_id)
            if len(user_rated_ij) == 0 :
                continue
            numerator = 0.0
            denominator_1 = 0.0
            denominator_2 = 0.0
            for user_id in range(n_users) :
                r_u_bar = np.mean(list(filter(lambda x : x!= 0, r_ui[user_id].tolist())))
                numerator += (r_ui[user_id][item_i] - r_u_bar) * (r_ui[user_id][item_j] - r_u_bar)
                denominator_1 += (r_ui[user_id][item_i] - r_u_bar)**2
                denominator_2 += (r_ui[user_id][item_j] - r_u_bar)**2
            p_ij[item_i][item_j] = numerator / (math.sqrt(denominator_1)*math.sqrt(denominator_2))

    n_ij = np.matmul(n_ui.T, n_ui)
    s_ij = np.multiply(np.divide(n_ij, np.add(n_ij, lambda_2)), p_ij)

    r_k_ui = [[[] for _ in range(n_users)] for _ in range(n_items)]
    n_k_ui = [[[] for _ in range(n_users)] for _ in range(n_items)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :

            user_rated = list(filter(lambda x : r_ui[x][item_id] != 0, [i for i in range(n_items)]))
            user_clicked = list(filter(lambda x : n_ui[x][item_id] != 0, [i for i in range(n_items)]))
            k_neighbors = list(sorted([i for i in range(n_items)], key=lambda x : -s_ij[item_id][x]))[:neighbor_size]

            user_rated = set(user_rated)
            user_clicked = set(user_clicked)
            k_neighbors = set(k_neighbors)

            r_k_ui[user_id][item_id] = list(user_rated.union(k_neighbors))
            n_k_ui[user_id][item_id] = list(user_clicked.union(k_neighbors))

    lr_all = 0.005
    lr_b_u = 0.007
    lr_b_i = 0.007
    lr_q_i = 0.007
    lr_p_u = 0.007
    lr_y_j = 0.007
    lr_w_ij = 0.001
    lr_c_ij = 0.001

    reg_all = 0.02
    reg_b_u = 0.005
    reg_b_i = 0.005
    reg_q_i = 0.015
    reg_p_u = 0.015
    reg_y_j = 0.015
    reg_w_ij = 0.015
    reg_c_ij = 0.015

    max_episode = 100
    performance_list = []

    for episode in range(max_episode) :

        b_ui = np.zeros((n_users, n_items)) + np.reshape(b_u, (1, -1)).T + b_i

        target_value = 0
        for user_id, item_id, rating in list(train_df.values) :

            w1_ui = np.zeros(n_factors)
            for item_j in n_u[user_id-1] :
                w1_ui += y_j[item_j-1]
            w1_ui /= math.sqrt(len(n_u[user_id-1]))

            w2_ui = 0
            for item_j in r_k_ui[user_id-1][item_id-1] :
                w2_ui += (r_ui[user_id-1][item_j] - b_ui[user_id-1][item_id-1]) * w_ij[item_id-1][item_j]
            w2_ui /= math.sqrt(len(r_k_ui[user_id-1][item_id-1]))

            w3_ui = 0
            for item_j in n_k_ui[user_id-1][item_id-1] :
                w3_ui += c_ij[user_id-1][item_id-1]
            w3_ui /= math.sqrt(len(n_k_ui[user_id-1][item_id-1]))

            r_hat_ui = b_ui[user_id-1][item_id-1] + np.dot(q_i[item_id-1], p_u[user_id-1] + w1_ui) + w2_ui + w3_ui

            err_ui = rating - r_hat_ui
            target_value += err_ui**2

            # Update
            b_u[user_id-1] += lr_b_u*(err_ui - reg_b_u*b_u[user_id-1])
            b_i[item_id-1] += lr_b_i*(err_ui - reg_b_i*b_i[item_id-1])
            q_i[item_id-1] += lr_q_i*(err_ui*w1_ui - reg_q_i*q_i[item_id-1])
            p_u[user_id-1] += lr_p_u*(err_ui*q_i[item_id-1] - reg_p_u*p_u[user_id-1])

            for item_j in n_ui[user_id-1] :
                y_j[item_j] += lr_y_j*((err_ui/math.sqrt(len(n_ui[user_id-1])))*q_i[item_id-1] - reg_y_j*y_j[item_j])

            for item_j in r_k_ui[user_id-1][item_id-1] :
                w_ij[item_id-1][item_j] += lr_w_ij*((err_ui*(r_ui[user_id-1][item_j] - b_ui[user_id-1][item_id-1])) / math.sqrt(len(r_k_ui[user_id-1][item_id-1])) 
                                                - reg_w_ij*w_ij[item_id-1][item_j])
            
            for item_j in n_k_ui[user_id-1][item_id-1] :
                c_ij[item_id-1][item_j] += lr_c_ij*((err_ui / math.sqrt(len(n_k_ui[user_id-1][item_id-1])) 
                                                - reg_c_ij*c_ij[item_id-1][item_j]))
        
        print(episode+1, target_value)

    # Test
    test_df, n_users, n_items = get_data(test_data)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :

        w1_ui = np.zeros(n_factors)
        for item_j in n_u[user_id-1] :
            w1_ui += y_j[item_j-1]
        w1_ui /= math.sqrt(len(n_u[user_id-1]))

        w2_ui = 0
        for item_j in r_k_ui[user_id-1][item_id-1] :
            w2_ui += (r_ui[user_id-1][item_j] - b_ui[user_id-1][item_id-1]) * w_ij[item_id-1][item_j]
        w2_ui /= math.sqrt(len(r_k_ui[user_id-1][item_id-1]))

        w3_ui = 0
        for item_j in n_k_ui[user_id-1][item_id-1] :
            w3_ui += c_ij[user_id-1][item_id-1]
        w3_ui /= math.sqrt(len(n_k_ui[user_id-1][item_id-1]))

        r_hat_ui = b_ui[user_id-1][item_id-1] + np.dot(q_i[item_id-1], p_u[user_id-1] + w1_ui) + w2_ui + w3_ui

        rmse += (rating - r_hat_ui)**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.4875874731082927


if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    n_factors = 10
    neighbor_size = 20
    integrated(train_data, test_data, n_factors, neighbor_size)
