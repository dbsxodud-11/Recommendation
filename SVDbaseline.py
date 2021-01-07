import numpy as np
import pandas as pd
import math
from utils import *

def SVDbaseline(train_data, test_data, n_factors) :

    train_df, n_users, n_items = get_data(train_data)

    # Latent Factor Model induced by Single Value Decomposition    
    r_ij = [[0 for _ in range(n_items)] for _ in range(n_users)]
    for user_id, item_id, rating in list(train_df.values) :
        r_ij[user_id-1][item_id-1] = rating
    r_ij = np.array(r_ij)

    # Initialize Biases
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)

    # Initialize Factors
    u_bar = np.array(r_ij.mean(axis = 1)).reshape(-1, 1)
    i_bar = np.array(r_ij.mean(axis = 0)).reshape(-1, 1)
    
    q_u = [np.random.normal(0, 0.1, n_factors) for i in range(n_users)]
    q_u = np.array(q_u)
    p_i = [np.random.normal(0, 0.1, n_factors) for i in range(n_items)]
    p_i = np.array(p_i)

    mu = np.mean(list(train_df.rating))

    # Use Gradient Descent Method
    max_episode = 50

    lr = 0.005
    lambda_3 = 0.02
    performance_list = []

    for episode in range(max_episode) :

        target_value = 0
        for user_id, item_id, rating in list(train_df.values) :

            r_hat_ui = mu + b_u[user_id-1] + b_i[item_id-1] + np.dot(q_u[user_id-1], p_i[item_id-1])

            err_ui = rating - r_hat_ui
            target_value += (err_ui)**2

            # Update
            b_u[user_id-1] += lr*(err_ui - lambda_3*b_u[user_id-1])
            b_i[item_id-1] += lr*(err_ui - lambda_3*b_i[item_id-1])
            q_u[user_id-1] += lr*(err_ui*p_i[item_id-1] - lambda_3*q_u[user_id-1])
            p_i[item_id-1] += lr*(err_ui*q_u[user_id-1] - lambda_3*p_i[item_id-1])

        print(episode+1, target_value)

    # Test
    test_df, n_users, n_items = get_data(test_data)

    # RMSE : test metric
    rmse = 0
    size = len(test_df.values)
    for user_id, item_id, rating in list(test_df.values) :
        rmse += (rating - (mu + b_u[user_id-1] + b_i[item_id-1] + np.dot(q_u[user_id-1], p_i[item_id-1])))**2
    rmse = math.sqrt(rmse / size)
    print(rmse) # 0.4875874731082927

if __name__ == "__main__" :

    train_data = "./ml-100k/sample.data"
    test_data = "./ml-100k/sample.data"
    n_factors = 100
    SVDbaseline(train_data, test_data, n_factors)
