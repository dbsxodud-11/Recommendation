import numpy as np
import pandas as pd
import math
from utils import *

def new_neighborhood_model(train_data, test_data, neighbor_size, implicit = False) :

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

    # Calculate Item Similarity
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

    w_ij = np.zeros((n_items, n_items))

    r_k_ui = [[[] for _ in range(n_users)] for _ in range(n_items)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :

            user_rated = list(filter(lambda x : r_ui[x][item_id] != 0, [i for i in range(n_items)]))
            k_neighbors = list(sorted([i for i in range(n_items)], key=lambda x : -s_ij[item_id][x]))[:neighbor_size]

            user_rated = set(user_rated)
            k_neighbors = set(k_neighbors)

            r_k_ui[user_id][item_id] = list(user_rated.union(k_neighbors))

    # Use Gradient Descent Method
    max_episode = 50

    lr = 0.005
    lambda_4 = 0.02
    performance_list = []

    for episode in range(max_episode) :

        target_value = 0
        for user_id, item_id, rating in list(train_df.values) :
            
            size = len(r_k_ui[user_id-1][item_id-1])
            if size == 0 :
                continue

            w_ui = 0
            b_ui = mu + b_u[user_id-1] + b_i[item_j]
            for item_j in r_k_ui[user_id-1][item_id-1] :
                w_ui += (r_ui[user_id-1][item_j] - b_ui) * w_ij[item_id-1][item_j]
            w_ui /= math.sqrt(size)

            if implicit :
                pass

            r_hat_ui = mu + b_u[user_id-1] + b_i[item_id-1] + w_ui

            err_ui = rating - r_hat_ui
            target_value += err_ui**2

            # Update
            b_u[user_id-1] += lr*(err_ui - lambda_4*b_u[user_id-1])
            b_i[item_id-1] += lr*(err_ui - lambda_4*b_i[item_id-1])

            for item_j in r_k_ui[user_id-1][item_id-1] :
                w_ij[item_id-1][item_j] += lr*((err_ui*(r_ui[user_id-1][item_j] - b_ui)) / math.sqrt(size) 
                                                - lambda_4*w_ij[item_id-1][item_j])
        
        print(episode+1, target_value)

    # Test
    test_df, n_users, n_items = get_data(test_data)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        w_ui = 0
        if len(r_k_ui[user_id-1][item_id-1]) != 0 :
            for item_j in r_k_ui[user_id-1][item_id-1] :
                w_ui += (r_ui[user_id-1][item_j] - (mu + b_u[user_id-1] + b_i[item_id-1]))*w_ij[item_id-1][item_j]
            w_ui /= len(r_k_ui)

        if implicit :
            pass

        rmse += (rating - (mu + b_u[user_id-1] + b_i[item_id-1] + w_ui))**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.7572667204589187

if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    neighbor_size = 20

    new_neighborhood_model(train_data, test_data, neighbor_size)