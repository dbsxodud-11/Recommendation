from data_structure.DataDAO import *
from data_structure.DataSplitter import *
from data_structure.SparseMatrix import *
from data_structure.SparseVector import *
from interfaces.Recommender import *

class GlobalAvg(Recommender) :

    def __init__(self, trainMatrix, testMatrix, fold) :
        super(GlobalAvg, self).__init__(trainMatrix, testMatrix, fold)

        self.algo_name = "GlobalAvg"
        # global mean
        numRates = self.trainMatrix.size()
        self.global_mean = self.trainMatrix.sum() / numRates
    
    def predict(self, u, i, rating=True) :
        return self.global_mean