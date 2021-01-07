import numpy as np
import pandas as pd
from utils import *
import math
import matplotlib.pyplot as plt

def baseline_estimates(train_data, test_data) :

    train_df, n_users, n_items = get_data(train_data)

    # baseline estimates using gradient descent
    train_avg_rating = np.mean(train_df.rating.values)

    ui_matrix = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        ui_matrix[user_id-1][item_id-1] = rating
    ui_matrix = np.array(ui_matrix)

    user_bias = np.zeros(n_users)
    item_bias = np.zeros(n_items)

    lr = 0.01
    lambda_1 = 0.1
    max_episode = 50
    performance_list = []

    # Training
    for episode in range(max_episode) :

        target_value = 0
        for user_id, item_id, rating in list(train_df.values) :
            error_ui = ui_matrix[user_id-1][item_id-1] - (train_avg_rating + user_bias[user_id-1] + item_bias[item_id-1])

            user_bias[user_id-1] += lr*(error_ui - lambda_1*user_bias[user_id-1])
            item_bias[item_id-1] += lr*(error_ui - lambda_1*item_bias[item_id-1])

            target_value += error_ui**2

        performance_list.append(target_value)
        print(episode+1, target_value)

    plt.plot(performance_list, label = "baseline estimates")
    plt.savefig("Collaborative_Filtering/plots/baseline_estimates(train).png")

    # Test
    test_df, n_users, n_items = get_data(test_data)
    test_avg_rating = np.mean(test_df.rating.values)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - (test_avg_rating + user_bias[user_id-1] + item_bias[item_id-1]))**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.9577214812556744

    # Store baseline estiamtes
    f = open("data/baseline_estimates.txt", "w")

    for i in range(n_users) :
        f.write(f"user_bias{i} : {user_bias[i]}\n")
    f.write("Item Bias")
    for i in range(n_items) :
        f.write(f"item_bias{i} : {item_bias[i]}\n")
    f.close()
    



if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    baseline_estimates(train_data, test_data)
