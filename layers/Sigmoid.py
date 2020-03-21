import numpy as np
from . import Dcg

class sigmoid:
    '''
    sigmoid activation fuction
    '''
    def __call__(self, *args, **kwargs):
        self.dcg = Dcg.DCG.getDCG()
        return self.forward(args[0])

    def sigmoid(self, x):
        '''
        forward adds dcg nodes so i split forward & sigmoid method
        '''
        return 1 / (1 + np.exp(-x))

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = Dcg.node(np.array(data))
        tmp.function = self.backward
        self.dcg.append(tmp)
        return self.sigmoid(tmp.data)

    def backward(self, input, gradient):
        gradient = self.sigmoid(input) * (1 - self.sigmoid(input)) * gradient
        return gradient