import numpy as np
from . import Dcg

class dropout:
    '''
    make mask at forward and save in dcg's data
    '''
    def __init__(self, ratio):
        self.ratio = ratio
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def forward(self, input):
        mask = np.random.randint(0, 99, input.shape)
        mask[mask < self.ratio*100] = 0
        mask[mask != 0] = 1
        mask = mask * input

        tmp = Dcg.node(mask)
        tmp.function = self.backward
        self.dcg.append(tmp)
        return mask

    def backward(self, mask, gradient):
        return mask * gradient