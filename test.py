'''
User guide
0. import nn.py
1. define model. In this example, 'net' class will be out model.
2. define layers and activation functions.
   You should define each Convolution, Linear layer but just define activation function, pooling layer once.
3. define forward method. Make your own feed forwarding.
4. load data and make instance of loss function.
5. use backward() in loss function's instance
'''
import pickle, gzip
import numpy as np
from tqdm import *
from layers.Conv2d import conv2d
from layers.Linear import Linear
from layers.Max_pool2d import max_pool2d
from layers.Sigmoid import sigmoid
from layers.Relu import relu
from layers.MSE import MSE
from layers.Dcg import zero_grad

# build model
class net:
    def __init__(self):
        self.conv1 = conv2d(1, 1, 3)
        self.fc1 = Linear(14*14, 10)
        self.max_pool = max_pool2d(2, 2)
        self.relu = relu()
        self.sigmoid = sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.sigmoid(self.fc1(x))
        return x

# data loading
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# train model
model = net()
for epoch in range(100):
    zero_grad()
    indexes = np.random.permutation(len(train_set[1]))
    for index in tqdm(indexes):
        img = np.reshape(train_set[0][index], (-1, 28, 28))  # input data reshaping
        label = np.zeros(10)  # convert to one-hot vector (maybe data loader can do this)
        label[train_set[1][index]] = 1
        output = model.forward(img)
        loss = MSE(output, label)
        loss.backward()
        #optimizer.step() <- not yet

    # valid model
    acc = 0
    for i in range(len(valid_set[1])):
        img = np.reshape(valid_set[0][i], (-1, 28, 28))  # input data reshaping
        output = model.forward(img)
        if output.argmax() == valid_set[1][i]:
            acc += 1
    print("epoch:{0}, accuracy:{1}".format(epoch, acc/len(valid_set[1])))

# output result
zero_grad()
acc = 0
for i in range(len(test_set[1])):
    img = np.reshape(test_set[0][i], (-1, 28, 28))  # input data reshaping
    output = model.forward(img)
    if output.argmax() == test_set[1][i]:
        acc += 1
print("accuracy:", acc/len(test_set[1]))