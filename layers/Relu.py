import numpy as np
from . import Dcg

class relu:
    '''
    ReLU activation function
    '''
    def __call__(self, *args, **kwargs):
        self.dcg = Dcg.DCG.getDCG()
        return self.forward(args[0])

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = Dcg.node(data)
        tmp.function = self.backward
        self.dcg.append(tmp)
        return np.maximum(0, data)

    def backward(self, input, gradient):
        '''
        using numpy boolean indexing(making mask)
        '''
        mask = np.maximum(0, input)
        mask[mask > 0] = 1
        mask = gradient * mask
        return mask