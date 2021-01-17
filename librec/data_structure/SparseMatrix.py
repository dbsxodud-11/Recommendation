
from bisect import bisect_left, bisect_right

class SparseMatrix :

    def __init__(self, rows, cols, dataTable, colMap, rowMap) :

        self.n_rows = rows
        self.n_cols = cols

        self.construct(dataTable, colMap, rowMap)

    def construct(self, dataTable, colMap, rowMap) :

        n_table = len(dataTable)

        # CRS
        self.rowPtr = [0 for _ in range(self.n_rows+1)]
        self.colIdx = [0 for _ in range(n_table)]
        self.rowData = [0.0 for _ in range(n_table)]

        j = 0
        for i in range(self.n_rows) :
            cols = list(sorted(rowMap.get(i)))
            self.rowPtr[i+1] = self.rowPtr[i] + len(cols)

            for col in cols :
                self.colIdx[j] = col
                j += 1
        
        # CCS
        self.colPtr = [0 for _ in range(self.n_cols+1)]
        self.rowIdx = [0 for _ in range(n_table)]
        self.colData = [0.0 for _ in range(n_table)]

        j = 0
        for i in range(self.n_cols) :
            rows = list(sorted(colMap.get(i)))
            self.colPtr[i+1] = self.colPtr[i] + len(rows)

            for row in rows :
                self.rowIdx[j] = row
                j += 1

        # Set Data
        for row, col, val in dataTable :
            r_index = self.getCRSIndex(row, col)
            self.rowData[r_index] = val

            c_index = self.getCCSIndex(row, col)
            self.colData[c_index] = val


    def getCRSIndex(self, row, col) :

        idx = bisect_left(self.colIdx, col, self.rowPtr[row], self.rowPtr[row+1])
        return idx

    def getCCSIndex(self, row, col) :

        idx = bisect_left(self.rowIdx, row, self.colPtr[col], self.colPtr[col+1])
        return idx