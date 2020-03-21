class node:
    '''
    class for store data and gradient
    data : layer's input data
    function : layer's back prapagation function
    '''
    def __init__(self, data=None):
        self.data = data
        self.function = None

class DCG:
    '''
    singleton class
    instance : singleton instance
    link : simple implementation of 'Dynamic Computational Graph' in linked list
    '''
    _instance = None

    def __init__(self):
        self.link = []

    @classmethod
    def getDCG(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def append(self, item):
        self.link.append(item)

    def pop(self):
        if len(self.link) > 0:
            return self.link.pop()
        else:
            return None

    def len(self):
        return len(self.link)

def zero_grad():
    dcg = DCG.getDCG()
    dcg.link.clear()