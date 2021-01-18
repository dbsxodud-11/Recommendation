

class SparseVector :

    def __init__(self, length) :
        self.length = length
        self.data = [0.0 for _ in range(self.length)]
    
    def getIndex(self) :
        return list(filter(lambda x : self.data[x]!=0.0, [i for i in range(self.length)]))

    def setValue(self, idx, val) :

        assert idx >= 0 and idx < self.length, "Index Error"
        self.data[idx] = val