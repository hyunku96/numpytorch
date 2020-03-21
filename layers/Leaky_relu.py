import numpy as np
from . import Dcg

class Leaky_relu:
    '''
    ReLU activation function
    '''
    def __init__(self, alpha):
        self.alpha = alpha
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = Dcg.node(data)
        tmp.function = self.backward
        self.dcg.append(tmp)
        return np.maximum(data * self.alpha, data)

    def backward(self, input, gradient):
        '''
        using numpy boolean indexing(making mask)
        '''
        mask = input
        mask[mask > 0] = 1
        mask[mask < 0] = self.alpha
        mask = gradient * mask
        return mask