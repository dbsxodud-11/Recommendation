from data_structure.SparseVector import *
from data_structure.MatrixEntry import *

from bisect import bisect_left, bisect_right

class SparseMatrix :

    def __init__(self, rows, cols, dataTable, colMap, rowMap, DataDAO) :

        self.n_rows = rows
        self.n_cols = cols

        self.construct(dataTable, colMap, rowMap)
        self.DataDAO = DataDAO

    # Iteration
    def __iter__(self) :
        self.index = 0
        return self

    def __next__(self) :
        if self.index >= self.size() :
            raise StopIteration
        entry = self.getEntry(self.index)
        self.index += 1
        return entry

    def construct(self, dataTable, colMap, rowMap) :

        n_table = len(dataTable)

        # CRS(Compressed Row Storage)
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
        
        # CCS(Compressed Column Storage)
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
            self.setValue(row, col, val)

    def setValue(self, row, col, val) :

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

    def getRow(self, row) :

        sv = SparseVector(self.n_cols)

        if row < self.n_rows :
            for j in range(self.rowPtr[row], self.rowPtr[row+1]) :
                col = self.colIdx[j]
                val = self.getValue(row, col)
                if val != 0.0 :
                    sv.setValue(col, val)

        return sv

    def getValue(self, row, col) :

        idx = bisect_left(self.colIdx, col, self.rowPtr[row], self.rowPtr[row+1])
        if idx >= 0 :
            return self.rowData[idx]
        else :
            return 0.0

    def reshape(self) :

        n_table = len(self.rowData)

        # CRS(Compressed Row Storage)
        new_rowPtr = [0 for _ in range(self.n_rows+1)]
        new_colIdx = [0 for _ in range(n_table)]
        new_rowData = [0.0 for _ in range(n_table)]

        new_idx = 0
        for i in range(len(self.rowPtr)-1) :
            for j in range(self.rowPtr[i], self.rowPtr[i+1]) :
                val = self.rowData[j]
                col = self.colIdx[j]
                if val != 0.0 :
                    new_rowData[new_idx] = val
                    new_colIdx[new_idx] = col
                    new_idx += 1
            new_rowPtr[i+1] = new_idx
        
        self.rowPtr = new_rowPtr
        self.colIdx = new_colIdx
        self.rowData = new_rowData
        
        n_table = len(self.colData)

        # CCS(Compressed Column Storage)
        new_colPtr = [0 for _ in range(self.n_cols+1)]
        new_rowIdx = [0 for _ in range(n_table)]
        new_colData = [0.0 for _ in range(n_table)]

        new_idx = 0
        for i in range(len(self.colPtr)-1) :
            for j in range(self.colPtr[i], self.colPtr[i+1]) :
                val = self.colData[j]
                row = self.rowIdx[j]
                if val != 0.0 :
                    new_colData[new_idx] = val
                    new_rowIdx[new_idx] = row
                    new_idx += 1
            new_colPtr[i+1] = new_idx
        self.colPtr = new_colPtr
        self.rowIdx = new_rowIdx
        self.colData = new_colData

    def size(self) :
        assert len(self.rowData) == len(self.colData), "rowData and colData unmatched"
        return len(self.rowData)

    def sum(self) :
        assert sum(self.rowData) == sum(self.colData), "rowData and colData unmatched"
        return sum(self.rowData)

    def getEntry(self, idx) :
        
        row = bisect_left(self.rowPtr, idx)
        col = idx
        val = self.rowData[idx]
        return MatrixEntry(row, col, val)