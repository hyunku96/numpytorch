import numpy as np
from struct import *

def get_MNIST():
    '''
    get MNIST handwritten digits
    label is one-hot vector
    :return: TrainImg, TrainLabel, TestImg, TestLabel
    '''
    # 파일 읽기
    fp_train_image = open('./training_set/train-images.idx3-ubyte', 'rb')
    fp_train_label = open('./training_set/train-labels.idx1-ubyte', 'rb')
    fp_test_image = open('./test_set/t10k-images.idx3-ubyte', 'rb')
    fp_test_label = open('./test_set/t10k-labels.idx1-ubyte', 'rb')
    # read mnist and show numbers
    fp_train_image.read(16)  # read first 16 byte
    fp_train_label.read(8)  # 1바이트씩 읽음

    fp_test_image.read(16)  # read first 16 byte
    fp_test_label.read(8)  # 1바이트씩 읽음

    #다음 4개의 변수들은 Singleton 에서 관리 하던지, main 파이썬 파일로 가져가야 함
    TrainImg = []
    TestImg = []
    TrainLabel = []
    TestLabel = []

    # train data 저장
    while True:
        s = fp_train_image.read(784)  # 784 바이트씩 읽음
        label = fp_train_label.read(1)  # 1 바이트 씩 읽음

        if not s:
            break
        if not label:
            break
        # unpack
        num = np.zeros(10)
        num[int(label[0])] = 1
        img = np.reshape(unpack(len(s)*'B',s), (1, 28, 28))/255.0 # byte를 unsigned char 형식으로
        TrainImg.append(img)
        TrainLabel.append(num)

    TrainImg = np.array(TrainImg)

    # test data 저장
    while True:
        s = fp_test_image.read(784)  # 784 바이트씩 읽음
        label = fp_test_label.read(1)  # 1 바이트 씩 읽음

        if not s:
            break
        if not label:
            break

        # unpack
        num = np.zeros(10)
        num[int(label[0])] = 1
        img = np.reshape(unpack(len(s)*'B',s), (1, 28,28))/255.0  # byte를 unsigned char 형식으로
        TestImg.append(img)
        TestLabel.append(num)

    TestImg = np.array(TestImg)

    return TrainImg, TrainLabel, TestImg, TestLabel