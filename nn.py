import numpy as np

class node:
    '''
    data : data of feed forwarding
    gradient : gradient of back propagation
    '''
    def __init__(self, data=None):
        self.data = data
        self.gradient = None

class DCG:
    '''
    singleton class
    link : simple implementation of 'Dynamic Computational Graph' in linked list
    curr : pointer of current backward calculation
    '''
    instance = None

    def __init__(self):
        head = node()
        self.link = [head]
        self.curr = 0

    def getDCG(self):
        if self.instance is None:
            self.instance = DCG()
        return self.intance

    def append(self, node):
        self.link.append(node)
        self.curr += 1

    def pop(self):
        self.curr -= 1
        return self.link.pop()

class Linear(DCG):
    '''
    fully-connected layer
    '''
    def __new__(self, *args, **kwargs):
        return self.forward

    def __init__(self, input_node, output_node):
        super().__init__()
        self.dcg = super().getDCG()
        self.w = np.random.rand(input_node, output_node) - 0.5
        self.b = np.random.rand(output_node) - 0.5

    def forward(self, data):
        data = np.ravel(data, order='C')
        tmp = node(np.dot(data, self.w) + self.b)
        self.dcg.append(tmp)

    def backward(self):
        pass

