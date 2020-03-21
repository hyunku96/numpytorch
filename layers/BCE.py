import numpy as np
from . import Dcg

class BCE:
    '''
    Binary Closs Entropy loss function
    compute BCE and initiate back propagation
    '''
    def __init__(self, output, label):
        self.output = np.clip(output, 1e-10, 1-1e-10)
        self.label = label
        self.dcg = Dcg.DCG.getDCG()

    def backward(self):
        '''
        if len(dcg) <= 0:
            print("Cannot find computation to calculate gradient")
            return
        else:
            loss = (self.output - self.label) / (self.output * (1 - self.output))
            tmp = dcg.pop()
            gradient = tmp.function(tmp.data, loss)
            while len(dcg) > 0:
                tmp = dcg.pop()
                gradient = tmp.function(tmp.data, gradient)
        '''
        if self.dcg.len() <= 0:
            print("Cannot find computation to calculate gradient")
            return
        else:
            loss = (1 - self.label)/(1 - self.output) - self.label / self.output
            tmp = self.dcg.pop()
            gradient = tmp.function(tmp.data, loss)
            while self.dcg.len() > 0:
                tmp = self.dcg.pop()
                gradient = tmp.function(tmp.data, gradient)