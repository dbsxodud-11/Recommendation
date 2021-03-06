# convert u.data to csv file
import pandas as pd
import numpy as np

def get_data(data) :

    df = []

    f = open(data, "r")
    while True :
        line = f.readline()
        line = list(map(int, line[:-2].split("::")[:-1]))
        if not line : break
        df.append(line)

    f.close()
    df = pd.DataFrame(df, columns = ["user_id", "item_id", "rating"])

    # Other Information
    n_users = max(df["user_id"].values)+1
    n_items = max(df["item_id"].values)+1

    return df, n_users, n_items


def get_baseline_estimates() :

    user_bias = []
    item_bias = []

    f = open("data/baseline_estimates.txt")
    while True :
        line = f.readline()
        if not line :
            break
        if line.startswith("user") :
            user_bias.append(float(line.split()[-1][:-2]))
        else :
            item_bias.append(float(line.split()[-1][:-2]))

    user_bias = np.array(user_bias)
    item_bias = np.array(item_bias)

    return user_bias, item_bias

if __name__ == "__main__" :
    data = "./ml-100k/u.data"
    df = get_data(data)
    get_baseline_estimates()
