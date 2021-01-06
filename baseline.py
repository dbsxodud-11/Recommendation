import numpy as np
import pandas as pd
from utils import *
import math
import matplotlib.pyplot as plt

def baseline_estimates(train_data, test_data) :

    train_df, n_users, n_items = get_data(train_data)
    # baseline estimates using gradient descent
    avg_rating = np.mean(train_df.rating.values)

    ui_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]
    user_rated = [[] for _ in range(n_users)]
    item_rated = [[] for _ in range(n_items)]
    for user_id, item_id, rating in list(train_df.values) :
        ui_matrix[user_id-1][item_id-1] = rating
        user_rated[user_id-1].append(item_id-1)
        item_rated[item_id-1].append(user_id-1)
    ui_matrix = np.array(ui_matrix)

    user_coef = np.zeros(n_users)
    item_coef = np.zeros(n_items)

    lr = 0.01
    lambda_1 = 0.1
    max_episode = 10
    performance_list = []

    # Training
    for episode in range(max_episode) :

        user_w = [0 for _ in range(n_users)]
        for user_id in range(n_users) :
            for item_id in user_rated[user_id] :
                error_ui = ui_matrix[user_id][item_id] - (avg_rating + user_coef[user_id] + item_coef[item_id])

                user_coef[user_id] += lr*(error_ui - lambda_1*user_coef[user_id])
                item_coef[item_id] += lr*(error_ui - lambda_1*item_coef[item_id])

        target_value = 0
        for user_id in range(n_users) :
            for item_id in user_rated[user_id] :
                target_value += (ui_matrix[user_id][item_id] - (avg_rating + user_coef[user_id] + item_coef[item_id]))**2
        performance_list.append(target_value)
        print(episode+1, target_value)

    plt.plot(performance_list, label = "baseline estimates")
    plt.savefig("Collaborative_Filtering/plots/baseline_estimates(train).png")

    # Test
    test_df, n_users, n_items = get_data(test_data)

    pred_ui_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]

    avg_rating = np.mean(test_df.rating.values)
    for user_id in range(n_users) :
        for item_id in range(n_items) :
            pred_ui_matrix[user_id][item_id] = avg_rating + user_coef[user_id] + item_coef[item_id]

    # RMSE : test metric
    rmse = 0
    size = 0
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - pred_ui_matrix[user_id-1][item_id-1])**2
        size += 1
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.9578541111590759

    # Store baseline estiamtes
    f = open("data/baseline_estimates.txt", "w")

    for i in range(n_users) :
        f.write(f"user_bias{i} : {user_coef[i]}\n")
    f.write("Item Bias")
    for i in range(n_items) :
        f.write(f"item_bias{i} : {item_coef[i]}\n")
    f.close()
    



if __name__ == "__main__" :

    train_data = "./ml-100k/u1.base"
    test_data = "./ml-100k/u1.test"
    baseline_estimates(train_data, test_data)
