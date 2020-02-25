import pickle, gzip
import numpy as np
from tqdm import *
import nn

# build model
class net:
    def __init__(self):
        self.fc1 = nn.Linear(784, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = nn.sigmoid(self.fc1(x))
        x = nn.sigmoid(self.fc2(x))
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
        loss = nn.MSE(output, train_set[1][index])
        loss.backward()
        #optimizer.step() <- not yet
    acc = 0
    for i in range(len(valid_set[1])):
        output = model.forward(train_set[0][i])
        if output.argmax() == valid_set[1][i]:
            acc += 1
    print("epoch:{0}, accuracy:{1}".format(epoch, acc/len(valid_set[1])))

# output result
acc = 0
for i in range(len(test_set[1])):
    output = model.forward(test_set[0][i])
    if output.argmax() == test_set[1][i]:
        acc += 1
print("accuracy:", acc/len(test_set[1]))