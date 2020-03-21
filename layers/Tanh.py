import numpy as np
from . import Dcg

class tanh:
    '''
    sigmoid activation fuction
    '''

    def __call__(self, *args, **kwargs):
        self.dcg = Dcg.DCG.getDCG()
        return self.forward(args[0])

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = Dcg.node(np.array(data))
        tmp.function = self.backward
        self.dcg.append(tmp)
        return np.tanh(tmp.data)

    def backward(self, input, gradient):
        gradient = (1 - np.tanh(input)) * (1 + np.tanh(input)) * gradient
        return gradient