from data_structure.DataDAO import *
from data_structure.DataSplitter import *
from data_structure.SparseMatrix import *
from data_structure.SparseVector import *

import math

class Recommender :

    def __init__(self, trainMatrix, testMatrix, fold) :

        self.trainMatrix = trainMatrix
        self.testMatrix = testMatrix
        self.fold = fold

        # rating measures
        self.isRankingPred = False

    def execute(self) :
        
        # Training a model
        self.init_model()
        self.train_model()

        # Evaluate
        self.evaluate_model()

    def init_model(self) :
        pass

    def train_model(self) :
        pass

    def evaluate_model(self) :

        if self.isRankingPred :
            self.evalRankings()
        else :
            self.evalRatings()

    def evalRatings(self) :

        mae_list = []
        mse_list = []

        self.minRate = self.testMatrix.DataDAO.minRate
        self.maxRate = self.testMatrix.DataDAO.maxRate
        
        # Create Output File
        f = open(f"{self.testMatrix.DataDAO.getDataDirectory()}result.txt", "w")
        
        for entry in self.testMatrix :

            true_rate = entry.val

            u = entry.row
            i = entry.col
            pred_rate = self.predict(u, i, True)
            f.write(f"{self.trainMatrix.DataDAO.userRow.get(u)}::{self.trainMatrix.DataDAO.itemCol.get(i)}::{pred_rate}\n")

            mae_list.append(abs(true_rate - pred_rate))
            mse_list.append(abs(true_rate - pred_rate)**2)

        mae = sum(mae_list) / len(mae_list)
        mse = sum(mse_list) / len(mse_list)
        rmae = math.sqrt(mae)
        rmse = math.sqrt(mse)

        self.measures = {"MAE" : mae, "MSE" : mse, "RMAE" : rmae, "RMSE" : rmse}
        f.close()
        


    def predict(self, u, i, rating=True) :
        raise NotImplementedError

    