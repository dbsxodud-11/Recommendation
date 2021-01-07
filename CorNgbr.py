import numpy as np
import pandas as pd
from utils import *
import math
import matplotlib.pyplot as plt

def CorNgbr(train_data, test_data, neighbor_size) :

    train_df, n_users, n_items = get_data(train_data)
    user_bias, item_bias = get_baseline_estimates()
    lambda_2 = 100

    # Implement CorNgbr
    avg_rating = np.mean(train_df.rating.values)

    ui_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]
    rated_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        ui_matrix[user_id-1][item_id-1] = rating
        rated_matrix[user_id-1][item_id-1] = 1
    ui_matrix = np.array(ui_matrix)
    rated_matrix = np.array(rated_matrix)

    # Calculate Item Similarity
    p_ij = np.corrcoef(ui_matrix.T)
    n_ij = np.matmul(rated_matrix.T, rated_matrix)

    s_ij = np.multiply(np.divide(n_ij, np.add(lambda_2, n_ij)), p_ij)

    # Predict User's Preference
    b_u = np.array([user_bias for _ in range(n_items)])
    b_i = np.array([item_bias for _ in range(n_users)])
    b_ui = np.add(avg_rating, np.add(b_u.T, b_i))

    w_ui = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            # Top k nearest neighbors
            item_rated_by_user_id = list(filter(lambda x : x!=0, rated_matrix[user_id]))
            k_neighbors = list(sorted(item_rated_by_user_id, key=lambda x : s_ij[item_id][x]))[:20]

            if len(k_neighbors) == 0 :
                continue
            else :
                numerator = 0
                denominator = 0
                for neighbor in k_neighbors :
                    numerator += s_ij[item_id][neighbor] * (ui_matrix[user_id][neighbor] - b_ui[user_id][neighbor])
                    denominator += s_ij[item_id][neighbor]
                w_ui[user_id][item_id] = numerator / denominator

    for user_id in range(n_users) :
        for item_id in range(n_items) :
            if ui_matrix[user_id][item_id] == 0 :
                r_ui_hat = b_ui + w_ui

    # Test
    test_df, n_users, n_items = get_data(test_data)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - ui_matrix[user_id-1][item_id-1])**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.9578541111590759

if __name__ == "__main__" :

    train_data = "./ml-100k/u1.base"
    test_data = "./ml-100k/u1.test"
    neighbor_size = 20
    CorNgbr(train_data, test_data, neighbor_size)

    # x = np.array([[1, 2], [3, 4]])
    # y = np.array([[0.1, 0.5], [0.3, 0.4]])

    # z = np.divide(x, y)
    # print(z)