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
    user_rated = [[] for _ in range(n_users)]
    item_rated = [[] for _ in range(n_items)]
    for user_id, item_id, rating in list(train_df.values) :
        ui_matrix[user_id-1][item_id-1] = rating
        item_rated[item_id-1].append(user_id-1)
        user_rated[user_id-1].append(item_id-1)

    # Calculate User Similarity
    item_similarity = [[0 for _ in range(n_items)] for _ in range(n_items)]
    for i in range(n_items) :
        for j in range(n_items) :
            if i == j :
                continue
            else :
                user_both_rated = list(filter(lambda x : x in item_rated[j], item_rated[i]))
                if len(user_both_rated) <= 1 :
                    continue
                numerator = 0
                denominator_1 = 0
                denominator_2 = 0

                rating_i = []
                rating_j = []
                for user_id in user_both_rated :
              
                    user_id_avg = 0
                    for item in user_rated[user_id] :
                        user_id_avg += ui_matrix[user_id][item]
                    user_id_avg /= len(user_rated[user_id])

                    numerator += (ui_matrix[user_id][i] - user_id_avg) * (ui_matrix[user_id][j] - user_id_avg)
                    denominator_1 += (ui_matrix[user_id][i] - user_id_avg)**2
                    denominator_2 += (ui_matrix[user_id][j] - user_id_avg)**2

                pearson_correlation = numerator / (math.sqrt(denominator_1)*math.sqrt(denominator_2))
                item_similarity[i][j] = (len(user_both_rated)*pearson_correlation) / (len(user_both_rated) + lambda_2)
    
    # Test
    test_df, n_users, n_items = get_data(test_data)

    pred_ui_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]

    avg_rating = np.mean(test_df.rating.values)
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            
            user_id_rated = list(filter(lambda x : x!= 0, ui_matrix[item_id]))
            k_neighbors = list(sorted(user_id_rated, key=lambda x : -item_similarity[item_id][x]))[:neighbor_size]
            numerator = 0
            denominator = 0
            for neighbor in k_neighbor :
                numerator += item_similarity[item_id][neighbor]*(ui_matrix[user_id][neighbor])
                denominator += item_similarity[item_id][neighbor]
            
            pred_ui_matrix[user_id][item_id] = (avg_rating + user_bias[user_id] + item_bias[item_id]) + (numerator/denominator)

    # RMSE : test metric
    print("test start")
    rmse = 0
    size = 0
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - pred_ui_matrix[user_id-1][item_id-1])**2
        size += 1
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.9578541111590759



if __name__ == "__main__" :

    train_data = "./ml-100k/u1.base"
    test_data = "./ml-100k/u1.test"
    neighbor_size = 20
    CorNgbr(train_data, test_data, neighbor_size)