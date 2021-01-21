from algorithms.baseline.GlobalAvg import *
from algorithms.both_cases import *
from algorithms.item_ranking import *
from algorithms.rating_prediction import *

from data_structure.DataDAO import *
from data_structure.SparseMatrix import *
from data_structure.DataSplitter import *

from interfaces.Recommender import *

import os
import argparse
import configparser
import shutil

def parse_args() :

    default_config_dir = "./config/config_global_avg_0.8_0.2.ini"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument("--dataset-split", type=str, required=False,
                        default="0.8 0.2", help="split train, validation, test dataset")
    args = parser.parse_args()
    return args

class LibRec :

    def __init__(self) :
        self.isMeasureOnly = False   # only print measurements
        self.tempDirPath = "./results/"  # output directory path

    def execute(self, args) :

        self.cmdLine(args) # process librec arguments

        # self.preset(self.config)    # reset general settings          
        self.runAlgorithm()         # run a speicific algorithm

        filename = self.algo_name + "_" +  str(self.trainRatio)
        # if self.validRatio is not None :
        #     filename += "_" + str(self.validRatio)
        result_dir = self.tempDirPath + filename
        shutil.copy(f"{self.rateDAO.getDataDirectory()}result.txt", result_dir)

    def cmdLine(self, args) :

        config_dir = args.config_dir
        self.config = configparser.ConfigParser()
        self.config.read(config_dir)

        self.readData() # read input data from config file

        # split the training data into "train-test" or "train-validation-test"
        ratios = args.dataset_split.rstrip().split(" ")
        if len(ratios) == 2 : # train-test
            trainRatio = float(ratios[0])
            testRatio = float(ratios[1])
            assert trainRatio + testRatio == 1.0, "sum is not equal to 1"
        else :
            trainRatio = float(ratios[0])
            validRatio = float(ratios[1])
            testRatio = float(ratios[2])
            assert trainRatio + validRatio + testRatio == 1.0, "sum is not equal to 1"
            self.validRatio = validRatio
        self.trainRatio = trainRatio
        self.testRatio = testRatio

        data_splitter = DataSplitter(self.rateMatrix)
        self.data = []
        if len(ratios) == 2 :
            self.data = data_splitter.getRatio(self.trainRatio, self.testRatio)
        else :
            self.data = data_splitter.getRatio(self.trainRatio, self.testRatio, self.validRatio)

        # Write Matrix
        dirPath = self.rateDAO.getDataDirectory()
        self.writeMatrix(self.data[0], dirPath + "train.txt")

        if len(ratios) == 2 :
            self.writeMatrix(self.data[1], dirPath + "test.txt")
        else :
            self.writeMatrix(self.data[1], dirpath + "validation.txt")
            self.writeMatrix(self.data[2], dirPath + "test.txt")

    def readData(self) :

        self.rateDAO = DataDAO(self.config.get("DATASET_CONFIG", "ratings")) # access to rating data
        columns = self.config.get("DATASET_CONFIG", "columns_to_use").rstrip().split(" ")    # data columns to use
        threshold = int(self.config.get("DATASET_CONFIG", "rating_threshold"))

        self.rateMatrix = self.rateDAO.readData(columns, threshold)

    def runAlgorithm(self) :

        method = self.config.get("EVALUATE_CONFIG", "method")
        if method == "cv" : # cross-validation
            self.runCrossValidation()
        elif method == "loo" : # leave-one-out
            self.runLeaveOneOut()
        elif method == "test" : # use test-set
            assert len(self.data) == 2, "train and test set only"
        
        algorithm = self.getRecommender(self.data, -1)
        algorithm.execute()

        self.printEvalInfo(algorithm.measures)

    def getRecommender(self, data, fold) :

        self.algo_name = self.config.get("ALGORITHM_CONFIG", "name")
        
        # self.writeData(data[0], data[1], fold)

        print(f"Algorithm : {self.algo_name}")
        if self.algo_name == "global_avg" :
            return GlobalAvg(data[0], data[1], fold)
        elif self.algo_name == "user_knn" :
            return UserkNN(data[0], data[1], fold)

    def writeMatrix(self, data, path) :

        f = open(path, "w")
        for i in range(data.n_rows) :
            for j in range(data.rowPtr[i], data.rowPtr[i+1]) :
                user = self.rateDAO.userRow.get(i)
                item = self.rateDAO.itemCol.get(data.colIdx[j])
                val = data.rowData[j]
                f.write(f"{user}::{item}::{val}\n")
        f.close()

    def printEvalInfo(self, measures) :

        print(measures)

if __name__ == "__main__" :

    librec = LibRec()
    args = parse_args()
    librec.execute(args)


    