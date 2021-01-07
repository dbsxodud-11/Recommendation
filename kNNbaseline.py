import numpy as np
import pandas as pd
import math
from utils import *

def kNN_baseline(train_data, test_data, neighbor_size) :

    train_df, n_users, n_items = get_data(train_data)

    r_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    n_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        r_ui[user_id-1][item_id-1] = rating
        n_ui[user_id-1][item_id-1] = 1
    r_ui = np.array(r_ui)
    n_ui = np.array(n_ui)

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

    # Predict User Preference
    b_u, b_i = get_baseline_estimates()
    b_ui = np.zeros((n_users, n_items)) + np.reshape(b_u, (1, -1)).T + b_i

    w_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            
            # Top k Nearest Neighbors
            item_rated_by_user_id = list(filter(lambda x : r_ui[user_id][x]!= 0, [i for i in range(n_items)]))
            k_neighbors = list(sorted(item_rated_by_user_id, key=lambda x : -s_ij[item_id][x]))[:neighbor_size]

            if len(k_neighbors) == 0 :
                continue
            else :
                numerator = 0.0
                denominator = 0.0
                for neighbor in k_neighbors :
                    numerator += s_ij[item_id][neighbor] * (r_ui[user_id][neighbor] - b_ui[user_id][neighbor])
                    denominator += s_ij[item_id][neighbor]
                w_ui[user_id][item_id] = numerator / denominator
    w_ui = np.array(w_ui)

    # Test
    test_df, n_users, n_items = get_data(test_data)
    test_avg_rating = np.mean(test_df.rating.values)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - (b_ui[user_id-1][item_id-1] + w_ui[user_id-1][item_id-1]))**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.40821710233374997

if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    neighbor_size = 20

    kNN_baseline(train_data, test_data, neighbor_size)