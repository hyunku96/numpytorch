import numpy as np
from . import Dcg

class conv2d:
    '''
    convolution layer
    User should declare input channel, kernel num, kenel size, padding mode.
    There's two padding mode - same, none. Default mode is same padding.
    '''
    def __init__(self, input_channel, kernel_num, kernel_size, padding="same"):
        self.input_channel, self.kernel_num, self.kernel_size = input_channel, kernel_num, kernel_size
        self.erosion = int(self.kernel_size/2)
        if padding is "none":
            self.padding = 0
        else:
            self.padding = self.erosion
        self.k, self.b = None, None
        self.dcg = Dcg.DCG.getDCG()

    def __call__(self, *args, **kwargs):
        '''
        args[0].shape[0, 1, 2] : depth, height, width of input
        args[0] is 4 dimension when model is batch-mode
        self.k : layer's kernel - 4 dimension(kernel num, input depth, kernel height, kernel width)
        self.b : layer's bias - 2 dimension(kernel num, output width * output height)
        '''
        if self.k is None and self.b is None:
            self.k = np.random.rand(self.kernel_num, self.input_channel, self.kernel_size, self.kernel_size) - 0.5
            self.b = np.random.rand(self.kernel_num, (np.array(args[0]).shape[1] + self.padding*2 - self.erosion*2), (np.array(args[0]).shape[2] + self.padding*2 - self.erosion*2)) - 0.5
        return self.forward(args[0])

    def forward(self, input):
        '''
        input, output : depth , height, width 3 dimension
        Ravel kernel and its corresponding area of input. It is similar with im2col method.
        '''
        img = np.pad(input, pad_width=self.padding, constant_values=(0))[self.padding:-self.padding]
        tmp = Dcg.node(input)
        tmp.function = self.backward
        self.dcg.append(tmp)

        kernel_mat = []
        for n in range(len(self.k)):
            kernel_mat.append(np.ravel(self.k[n], order='C'))

        input_mat = []
        for height in range(self.erosion, img.shape[1] - self.erosion):
            for width in range(self.erosion, img.shape[2] - self.erosion):
                input_mat.append(np.ravel(img[:, height - self.erosion:height + self.erosion + 1, width - self.erosion:width + self.erosion + 1], order='C'))
        # send kernel_mat & kernel_mat to GPU!
        result = np.matmul(input_mat, np.array(kernel_mat).T)
        result = np.reshape(result, (self.kernel_num, (np.array(input).shape[1] + self.padding*2 - self.erosion*2), (np.array(input).shape[2] + self.padding*2 - self.erosion*2))) + self.b
        '''
        result = []
        for n in range(len(self.k)):  # kernel num
            out = []
            for height in range(self.erosion, img.shape[1] - self.erosion):  # input height
                for width in range(self.erosion, img.shape[2] - self.erosion):  # input width
                    area = img[:, height - self.erosion:height + self.erosion + 1, width - self.erosion:width + self.erosion + 1]
                    sum = np.sum(np.ravel(area, order='C') * np.ravel(self.k[n], order='C'))
                    out.append(sum)
            out = out + self.b[n]
            result.append(np.reshape(out, ((np.array(input).shape[1] + self.padding*2 - self.erosion*2), (np.array(input).shape[2] + self.padding*2 - self.erosion*2))))
        '''
        return result

    def backward(self, input, gradient):
        # calculate kernel's gradient
        img = np.pad(input, pad_width=self.padding, constant_values=(0))[self.padding:-self.padding]
        dk = np.array([])
        # convolution input & gradient to compute kernel's gradient
        for n in range(self.kernel_num):  # kernel num
            for depth in range(self.input_channel):  # input's depth
                for height in range(self.kernel_size):  # input's height
                    for width in range(self.kernel_size):  # input's width
                        area = img[depth][height:height + gradient.shape[1], width:width + gradient.shape[2]]
                        tmp = np.sum(np.ravel(area, order='C') * np.ravel(gradient[n], order='C'))
                        dk = np.append(dk, tmp)
        dk = np.reshape(dk, (self.kernel_num, self.input_channel, self.kernel_size, self.kernel_size))  # num, depth, height, width
        # convolution kernel(rotation 180') & gradient to compute input's gradient
        output = np.array([])
        for n in range(self.kernel_num):  # gradient's depth
            plank = np.pad(gradient[n], pad_width=self.padding, constant_values=(0))  # a slice of gradient
            _k = np.rot90(self.k[n], 2)  # 180' rotation

            tmp = []
            for depth in range(self.input_channel):  # kernel's depth
                for height in range(self.erosion, plank.shape[0] - self.erosion):  # kernel's height
                    for width in range(self.erosion, plank.shape[1] - self.erosion):  # kernel's width
                        area = plank[height - self.erosion:height + self.erosion + 1, width - self.erosion:width + self.erosion + 1]
                        tmp.append(np.sum(np.ravel(area, order='C') * np.ravel(_k[depth], order='C')))
            output = np.append(output, tmp)

        output = np.reshape(output, (self.kernel_num, self.input_channel, np.array(input).shape[1], np.array(input).shape[2]))
        output = output.sum(axis=0) / self.kernel_num  # mean of each kernel's gradient

        # update weights(change lr in optim class)
        self.k = self.k - 0.01 * dk
        self.b = self.b - 0.01 * gradient

        return output