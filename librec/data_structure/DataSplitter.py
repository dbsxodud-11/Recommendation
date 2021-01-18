from data_structure.SparseVector import *

import copy
import random

class DataSplitter :

    def __init__(self, rateMatrix) :

        self.rateMatrix = rateMatrix

    def getRatio(self, trainRatio, testRatio, validRatio=0.0) :

        # getRatio by rating only
        trainMatrix = copy.deepcopy(self.rateMatrix)
        testMatrix = copy.deepcopy(self.rateMatrix)

        for user in range(self.rateMatrix.n_rows) :
            
            user_v = self.rateMatrix.getRow(user)
            for item in user_v.getIndex() :
                if random.random() < trainRatio :
                    testMatrix.setValue(user, item, 0.0)
                else :
                    trainMatrix.setValue(user, item, 0.0)

        trainMatrix.reshape()
        testMatrix.reshape()

        return trainMatrix, testMatrix




