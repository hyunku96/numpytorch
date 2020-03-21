import numpy as np
from . import Dcg

class Linear:
    '''
    fully-connected layer
    '''
    def __init__(self, input_node, output_node):
        '''
        get instance of DCG and init pramters
        '''
        self.w = np.random.rand(input_node, output_node) - 0.5
        self.b = np.random.rand(output_node) - 0.5
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0] is 2 dimension when model is batch-mode
        '''
        return self.forward(args)

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        data = np.reshape(data, (1, -1))
        tmp = Dcg.node(data)
        tmp.function = self.backward
        self.dcg.append(tmp)
        output = np.dot(data, self.w) + self.b
        return output

    def backward(self, input, gradient):
        dw = np.dot(np.reshape(input, (-1, 1)), gradient)
        db = gradient
        gradient = np.dot(gradient, self.w.T)
        self.w = self.w - 0.01 * dw  # change lr in optimizer class
        self.b = self.b - 0.01 * db
        return gradient
