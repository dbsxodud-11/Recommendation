from data_structure.SparseMatrix import SparseMatrix

from collections import defaultdict

class DataDAO :

    def __init__(self, rating_path, userIds={}, itemIds={}) :

        self.path = rating_path
        if "ml-100k" in self.path :
            self.data_name = "MovieLens-100k"
        elif "ml-1m" in self.path :
            self.data_name = "MovieLens-1M"
        self.scaleDist = set()
        self.userIds = userIds
        self.itemIds = itemIds

    def readData(self, columns, threshold) :

        print(f"DataSet: {self.data_name}")
        dataTable = set() # {row_id, col-id, rate}
        rowMap = defaultdict(lambda: []) # {row_id, multiple col_id}
        colMap = defaultdict(lambda: []) # {col_id, multiple row_id}

        with open(self.path, "r") as f :
            lines = f.readlines()
            for line in lines :
                line = list(map(int, line.rstrip().split("::")))
                user = line[0]
                item = line[1]
                rating = line[2]
                if threshold > rating :
                    rating = threshold

                self.scaleDist.add(rating)
                row = 0
                if user not in self.userIds :
                    row = len(self.userIds)
                    self.userIds[user] = row
                else :
                    row = self.userIds.get(user)
                col = 0
                if item not in self.itemIds :
                    col = len(self.itemIds)
                    self.itemIds[item] = col
                else :
                    col = self.itemIds.get(item)
                
                dataTable.add((row, col, rating))
                colMap[col].append(row)
                rowMap[row].append(col)
        
        numRatings = len(self.scaleDist)
        ratingScale = list(self.scaleDist)
        ratingScale.sort()

        # print(dataTable, self.userIds, self.itemIds, colMap, rowMap)

        n_users = len(self.userIds)
        n_items = len(self.itemIds)
        rateMatrix = SparseMatrix(n_users, n_items, dataTable, colMap, rowMap)

        return rateMatrix

