from algorithms.baseline import *
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
        isMeasureOnly = False   # only print measurements
        tempDirPath = "./Results/"  # output directory path

    def execute(self, args) :

        self.cmdLine(args) # process librec arguments

        self.preset(self.config)    # reset general settings          
        self.runAlgorithm()         # run a speicific algorithm

        filename = self.algorithm + "_" +  str(self.trainRatio) + "_"
        if self.validRatio is not None :
            filename += "_" + str(self.validRatio)
        result_dir = tempDirPath + filename
        shutil.copy("result.txt", result_dir)

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

        data_splitter = DataSplitter(self.rateMatrix)
        self.data = []
        if len(ratios) == 2 :
            self.data = data_splitter.getRatio(trainRatio, testRatio)
        else :
            self.data = data_splitter.getRatio(trainRatio, validRatio, testRatio)
        
        # Write Matrix
        dirPath = self.rateDAO.getDataDirectory()
        self.writeMatrix(self.data[0], dirPath + "train.txt")

        if len(ratios) == 2 :
            self.writeMatrix(self.data[1], dirpath + "test.txt")
        else :
            self.writeMatrix(self.data[1], dirpath + "validation.txt")
            self.writeMatrix(self.data[2], dirPath + "test.txt")

    def readData(self) :

        rateDAO = DataDAO(self.config.get("DATASET_CONFIG", "ratings")) # access to rating data
        columns = self.config.get("DATASET_CONFIG", "columns_to_use").rstrip().split(" ")    # data columns to use
        threshold = int(self.config.get("DATASET_CONFIG", "rating_threshold"))

        self.rateMatrix = rateDAO.readData(columns, threshold)
        
        self.Recommender = Recommender(self.rateMatrix, rateDAO, threshold)

    def runAlgorithm(self) :

        method = self.config["EVALUATE_CONFIG", "method"]
        if method == "cv" : # cross-validation
            self.runCrossValidation()
        elif method == "loo" : # leave-one-out
            self.runLeaveOneOut()
        elif method == "test" : # use test-set
            assert len(self.data) == 2, "train and test set only"
        
        algorithm = self.getRecommneder(self.data, -1)
        algorithm.execute()

        self.printEvalInfo(algorithm, algorithm.measures)


if __name__ == "__main__" :

    librec = LibRec()
    args = parse_args()
    librec.execute(args)


    