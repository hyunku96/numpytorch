import numpy as np
from . import Dcg


class max_pool2d:
    '''
    max pooling layer
    need to add input's number iteration
    '''

    def __init__(self, hsize, wsize):
        self.h_size, self.w_size = hsize, wsize
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0] : input data
        '''
        return self.forward(args[0])

    def forward(self, input):
        '''
        slicing every pooling area in input then find max value.
        Assemble max values and reshape it.
        '''

        tmp = Dcg.node(input)
        tmp.function = self.backward
        self.dcg.append(tmp)

        result = []
        for depth in range(input.shape[0]):
            out = []
            for height in range(0, input.shape[1] - 1, self.h_size):
                for width in range(0, input.shape[2] - 1, self.w_size):
                    out.append(np.max(input[depth, height:height + self.h_size, width:width + self.w_size]))
            result.append(np.reshape(out, (int(input.shape[1] / self.h_size), int(input.shape[2] / self.w_size))))

        return result


    def backward(self, input, gradient):
        gradient = np.reshape(gradient,
                              (input.shape[0], int(input.shape[1] / self.h_size), int(input.shape[2] / self.w_size)))
        mask = np.zeros((input.shape[0], input.shape[1], input.shape[2]))
        for depth in range(input.shape[0]):
            for height in range(0, input.shape[1], self.h_size):
                for width in range(0, input.shape[2], self.w_size):
                    area = input[depth][height:height + self.h_size, width:width + self.w_size]
                    if area.any():  # if area's elements were all zero(caused by relu), gradients shall not pass
                        maxloc = area.argmax()
                        # assign rear layer's gradient to maxpooling & ReLU layer's gradient
                        mask[depth, height + int(maxloc / self.h_size), width + maxloc % self.w_size] = gradient[
                            depth, int(height / self.h_size), int(width / self.w_size)]

        return mask