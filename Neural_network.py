#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Neural_Networks
@File    ：Neural_network.py
@Author  ：易鹏飞
@Content ：numpy神经网络完成mnist数据集上多分类任务
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Linear(object):
    """全连接层
    参数:
        n_in:int 输入维度
        n_out:int 输出维度
    """

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.input = None
        self.weight = None
        self.bias = None
        self.d_weight = None
        self.d_bias = None

    def __call__(self, input):
        return self.forward(input)

    def init_param(self, std=0.01):
        # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.n_in, self.n_out))
        self.bias = np.zeros([1, self.n_out])

    def forward(self, input):
        # 前向传播计算
        self.input = input
        output = np.matmul(input, self.weight) + self.bias
        return output

    def backward(self, preGrad):
        # 反向传播计算
        self.d_weight = np.dot(self.input.T, preGrad)
        self.d_bias = np.sum(preGrad, axis=0)
        postGrad = np.dot(preGrad, self.weight.T)
        return postGrad

    def update_param(self, lr):
        # 对全连接层参数利用参数进行更新(梯度下降)
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        # 参数加载
        self.weight = weight
        self.bias = bias

    def save_param(self):
        # 参数保存
        return self.weight, self.bias


class Relu(object):
    """Relu 激活函数
    """

    def __init__(self):
        self.input = None

    def forward(self, input):
        # Relu 前向传播的计算
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, preGrad):
        # Relu 反向传播的计算
        postGrad = preGrad
        postGrad[self.input < 0] = 0
        return postGrad


class Sigmoid(object):
    """sigmoid 激活函数
    """

    def __init__(self):
        self.output = None

    def forward(self, input):
        # sigmoid 前向传播的计算
        output = 1 / (1 + np.exp(-input))
        self.output = output
        return output

    def backward(self, preGrad):
        # sigmoid 反向传播的计算
        postGrad = preGrad * (1.0 - self.output) * self.output
        return postGrad


class SoftmaxWithLoss(object):
    """附带交叉熵损失的SoftMax
    """

    def __init__(self):
        self.prob = None
        self.batch_size = None
        self.label_onehot = None

    def forward(self, input):
        # 前向传播的计算
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label.astype(int)] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        # 反向传播的计算
        postGrad = (self.prob - self.label_onehot) / self.batch_size
        return postGrad


class MNIST_MLP(object):
    """mnist手写数字集多层感知机
    """

    def __init__(self, batch_size=30, input_size=64, hidden1=32, hidden2=16, out_size=10, lr=0.01,
                 max_epoch=30, print_iter=10):
        """
        参数初始化

        :param batch_size:
        :param input_size:
        :param hidden1:
        :param hidden2:
        :param out_size:
        :param lr:
        :param max_epoch:
        :param print_iter:
        """
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_size = out_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_data(self):
        # 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
        digits = datasets.load_digits()
        train_images, test_images, train_labels, test_labels \
            = train_test_split(digits.data, digits.target, test_size=0.2)
        # FIXME 查看 image 和 label 的形式
        self.train_data = np.append(train_images, train_labels.reshape(-1, 1), axis=1)
        self.test_data = np.append(test_images, test_labels.reshape(-1, 1), axis=1)

    def shuffle_data(self):
        # 打乱数据
        np.random.shuffle(self.train_data)

    def build_model(self):
        # 建立三层神经网络结构
        self.layer1 = Linear(self.input_size, self.hidden1)
        self.sigmoid1 = Sigmoid()
        self.layer2 = Linear(self.hidden1, self.hidden2)
        self.sigmoid2 = Sigmoid()
        self.layer3 = Linear(self.hidden2, self.out_size)
        self.softmax = SoftmaxWithLoss()
        self.layer_list = [self.layer1, self.layer2, self.layer3]

    def init_model(self):
        # 初始化多层感知机的全连接层
        for layer in self.layer_list:
            layer.init_param()

    def forward(self, input):
        # 神经网络的前向传播
        h1 = self.layer1.forward(input)
        h1 = self.sigmoid1.forward(h1)
        h2 = self.layer2.forward(h1)
        h2 = self.sigmoid2.forward(h2)
        h3 = self.layer3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh3 = self.layer3.backward(dloss)
        dh2 = self.sigmoid2.backward(dh3)
        dh2 = self.layer2.backward(dh2)
        dh1 = self.sigmoid1.backward(dh2)
        dh1 = self.layer1.backward(dh1)

    def update(self, lr):
        for layer in self.layer_list:
            layer.update_param(lr)

    def train(self):
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                # TODO 查看 labels 是否为列向量
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)


def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 256, 128, 1000
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    return mlp


if __name__ == '__main__':
    print(__doc__)
    mlp = build_mnist_mlp()
    mlp.evaluate()
