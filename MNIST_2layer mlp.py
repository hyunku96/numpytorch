'''
User guide
0. import nn.py
1. define model. In this example, 'net' class will be out model.
2. define layers and activation functions.
   You should define each Linear layer but just define activation functoin once.
3. define forward method. Make your own feed forwarding.
4. load data and make instance of loss function.
5. use backward() in loss function's instance
'''
import pickle, gzip
import numpy as np
from tqdm import *
import nn

# build model
class net:
    def __init__(self):
        self.fc1 = nn.Linear(784, 28)
        self.fc2 = nn.Linear(28, 10)
        self.relu = nn.relu()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# data loading
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# train model
model = net()
for epoch in range(100):
    indexes = np.random.permutation(len(train_set[1]))
    for index in tqdm(indexes):
        output = model.forward(train_set[0][index])
        label = np.zeros(10)  # convert to one-hot vector (maybe data loader can do this)
        label[train_set[1][index]] = 1
        loss = nn.MSE(output, label)
        loss.backward()
        #optimizer.step() <- not yet

    # valid model
    acc = 0
    for i in range(len(valid_set[1])):
        output = model.forward(valid_set[0][i])
        nn.dcg.clear()  # valid forwarding is stacking dcg so i should clear dcg
        if output.argmax() == valid_set[1][i]:
            acc += 1
    print("epoch:{0}, accuracy:{1}".format(epoch, acc/len(valid_set[1])))

# output result
acc = 0
for i in range(len(test_set[1])):
    output = model.forward(test_set[0][i])
    nn.dcg.clear()
    if output.argmax() == test_set[1][i]:
        acc += 1
print("accuracy:", acc/len(test_set[1]))