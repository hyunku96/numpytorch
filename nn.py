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
        return self.forward(args)

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        data = np.ravel(data, order='C')
        tmp = node(data)
        tmp.function = self.backward
        dcg.append(tmp)
        return np.dot(data, self.w) + self.b

    def backward(self, input, gradient, lr):
        dw = np.dot(input, gradient)
        db = gradient
        gradient = np.dot(self.w, gradient)
        self.w = self.w - lr * dw
        self.b = self.b - lr * db
        return gradient

class sigmoid:
    '''
    sigmoid activation fuction
    '''
    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = node(data)
        tmp.function = self.backward
        dcg.append(tmp)
        return 1 / (1 + np.exp(-np.asarray(data)))

    def backward(self, input):
        return self.forward(input) * (1 - self.forward(input))

class relu:
    '''
    ReLU activation function
    '''
    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def forward(self, data):
        '''
        feed forward and store input data and backward function to DCG
        '''
        tmp = node(data)
        tmp.function = self.backward
        dcg.append(tmp)
        return np.maximum(0, data)

    def backward(self, input):
        '''
        using numpy boolean indexing(mask)
        '''
        gradient = np.maximum(0, input)
        gradient[gradient > 0] = 1
        return gradient


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