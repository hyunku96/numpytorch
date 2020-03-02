'''
All layers are defined in class but 'Dynamic Computational Graph' is defined in list structure.
Loss function starts back propagation by using pop method of DCG.
There exists only Gradient Decent optimization method.
I will change model's batch size, DCG structure and make more optimization methods.
Data Loader should make input in 3 dimension and make label to one-hot vector.
'''
import numpy as np

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
    To Do - change list to graph for improve of DCG
    instance : singleton instance
    link : simple implementation of 'Dynamic Computational Graph' in linked list
    '''
    instance = None

    def __init__(self):
        self.link = []

    def getDCG(self):
        if self.instance is None:
            self.instance = DCG()
        return self.intance

    def append(self, node):
        self.link.append(node)

    def pop(self):
        if len(self.link) > 0:
            return self.link.pop()
        else:
            return None

dcg = []

def init_dcg():
    dcg.clear()

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
        tmp = node(data)
        tmp.function = self.backward
        dcg.append(tmp)
        output = np.dot(data, self.w) + self.b
        return output

    def backward(self, input, gradient):
        dw = np.dot(np.reshape(input, (-1, 1)), gradient)
        db = gradient
        gradient = np.dot(gradient, self.w.T)
        self.w = self.w - 0.01 * dw  # change lr in optimizer class
        self.b = self.b - 0.01 * db
        return gradient

class conv2d:
    '''
    convolution layer
    User should declare input channel, kernel num, kenel size, padding mode.
    There's two padding mode - same, none. Default mode is same padding.
    '''
    def __init__(self, input_channel, kernel_num, kernel_size, padding="same"):
        self.input_channel, self.kernel_num, self.kernel_size = input_channel, kernel_num, kernel_size
        if padding is "none":
            self.padding = 0
        else:
            self.padding = int(self.kernel_size / 2)

    def __call__(self, *args, **kwargs):
        '''
        args[0].shape[0, 1, 2] : depth, height, width
        args[0] is 4 dimension when model is batch-mode
        self.k : layer's kernel - 4 dimension(kernel num, input depth, kernel width, kernel height)
        self.b : layer's bias - 3 dimension(kernel num, output width, output height)
        '''
        self.k = np.random.rand(self.kernel_num, self.input_channel, self.kernel_size, self.kernel_size) - 0.5
        self.b = np.random.rand(self.kernel_num, np.array(args[0]).shape[1] * np.array(args[0]).shape[2])
        return self.forward(args[0])

    def forward(self, input):
        '''
        input, output : depth , height, width 3 dimension
        Ravel kernel and its corresponding area of input. It is similar with im2col method.
        '''
        result = []
        img = np.pad(input, pad_width=self.padding, constant_values=(0))[self.padding:-1]
        tmp = node(input)
        tmp.function = self.backward
        dcg.append(tmp)

        for n in range(len(self.k)):  # kernel num
            out = []
            for height in range(self.padding, img.shape[1] - self.padding):  # input height
                for width in range(self.padding, img.shape[2] - self.padding):  # input width
                    area = img[:, height - self.padding:height + self.padding + 1, width - self.padding:width + self.padding + 1]
                    sum = np.sum(np.ravel(area, order='C') * np.ravel(self.k[n], order='C'))
                    out.append(sum)
            out = out + self.b[n]
            result.append(np.reshape(out, (np.array(input).shape[1], np.array(input).shape[2])))

        return result

    def backward(self, input, gradient):
        # calculate kernel's gradient
        img = np.pad(input, pad_width=self.padding, constant_values=(0))[1:-1]
        dk = np.array([])
        # convolution input & gradient to compute kernel's gradient
        for n in range(self.kernel_num):  # kernel num
            for depth in range(self.input_channel):  # gradient's channel
                for height in range(self.kernel_size):  # height interval
                    for width in range(self.kernel_size):  # width interval
                        area = img[depth][height:height + gradient.shape[1], width:width + gradient.shape[2]]
                        tmp = np.sum(np.ravel(area, order='C') * np.ravel(gradient[n], order='C'))
                        dk = np.append(dk, tmp)
        dk = np.reshape(dk, (self.kernel_num, self.input_channel, self.kernel_size, self.kernel_size))  # num, depth, height, width
        # convolution kernel(rotation 180') & gradient to compute input's gradient
        output = np.array([])
        for n in range(self.kernel_num):  # kernel num
            plank = np.pad(gradient[n], pad_width=self.padding, constant_values=(0))  # a slice of gradient
            _k = np.rot90(self.k[n], 2)  # 180' rotation

            tmp = []
            for depth in range(self.input_channel):  # input's depth
                for height in range(1, plank.shape[0] - 1):
                    for width in range(1, plank.shape[1] - 1):
                        area = plank[height - 1:height + 2, width - 1:width + 2]
                        tmp.append(np.sum(np.ravel(area, order='C') * np.ravel(_k[depth], order='C')))
            output = np.append(output, tmp)

        output = np.reshape(output, (self.kernel_num, self.input_channel, np.array(input).shape[1], np.array(input).shape[2]))
        output = output.sum(axis=0) / self.kernel_num  # mean of each kernel's gradient

        # update weights(change lr in optim class)
        gradient = np.reshape(gradient, (self.kernel_num, -1))
        self.k -= 0.01 * dk
        self.b -= 0.01 * gradient

        return output

class max_pool2d:
    '''
    max pooling layer
    need to add input's number iteration
    '''
    def __init__(self, hsize, wsize):
        self.h_size, self.w_size = hsize, wsize

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
        tmp = node(input)
        tmp.function = self.backward
        dcg.append(tmp)

        result = []
        for depth in range(input.shape[0]):
            out = []
            for height in range(0, input.shape[1], self.h_size):
                for width in range(0, input.shape[2], self.w_size):
                    out.append(np.max(input[depth, height:height + self.h_size, width:width + self.w_size]))
            result.append(np.reshape(out, (int(input.shape[1]/2), int(input.shape[2]/2))))

        return result

    def backward(self, input, gradient):
        gradient = np.reshape(gradient, (input.shape[0], int(input.shape[1]/2), int(input.shape[2]/2)))
        mask = np.zeros((input.shape[0], input.shape[1], input.shape[2]))
        for depth in range(input.shape[0]):
            for height in range(0, input.shape[1], self.h_size):
                for width in range(0, input.shape[2], self.w_size):
                    area = input[depth][height:height + self.h_size, width:width + self.w_size]
                    if area.all() != 0:  # if area's elements were all zero, gradients shall not pass
                        maxloc = area.argmax()
                        # assign rear layer's gradient to maxpooling & ReLU layer's gradient
                        mask[depth, height + int(maxloc / 2), width + maxloc % 2] = gradient[depth, int(height / 2), int(width / 2)]

        return mask

class sigmoid:
    '''
    sigmoid activation fuction
    '''
    def __call__(self, *args, **kwargs):
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
        tmp = node(np.array(data))
        tmp.function = self.backward
        dcg.append(tmp)
        return self.sigmoid(tmp.data)

    def backward(self, input, gradient):
        gradient = self.sigmoid(input) * (1 - self.sigmoid(input)) * gradient
        return gradient

class relu:
    '''
    ReLU activation function
    '''
    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = node(data)
        tmp.function = self.backward
        dcg.append(tmp)
        return np.maximum(0, data)

    def backward(self, input, gradient):
        '''
        using numpy boolean indexing(making mask)
        '''
        mask = np.maximum(0, input)
        mask[mask > 0] = 1
        gradient = gradient * mask
        return gradient

class MSE:
    '''
    loss function
    compute MSE and initiate back propagation
    '''
    def __init__(self, output, label):
        self.output = output
        self.label = label

    def backward(self):
        if len(dcg) <= 0:
            print("Cannot find computation to calculate gradient")
            return
        else:
            loss = self.output - self.label
            tmp = dcg.pop()
            gradient = tmp.function(tmp.data, loss)
            while len(dcg) > 0:
                tmp = dcg.pop()
                gradient = tmp.function(tmp.data, gradient)

'''
class test:
    def __init__(self):
        tmp = node(0)
        tmp.function = self.backward
        dcg.append(tmp)

    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def forward(self, data):
        print(data)
        return data

    def backward(self):
        print("backward")

t = test()
print(t(1, 2, 3))
dcg.pop().function()
'''