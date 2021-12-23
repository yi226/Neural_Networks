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
import matplotlib.pyplot as plt


class MNIST(object):
    """mnist手写数字集
    """

    def __call__(self):
        return self.load_data()

    @staticmethod
    def load_data():
        # 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
        digits = datasets.load_digits()
        train_images, test_images, train_labels, test_labels \
            = train_test_split(digits.data, digits.target, test_size=0.2)
        train_data = np.append(train_images, train_labels.reshape(-1, 1), axis=1)
        test_data = np.append(test_images, test_labels.reshape(-1, 1), axis=1)
        return train_data, test_data


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
        self.params = {}
        self.d_params = {}
        self.pre = None

    def __call__(self, input):
        return self.forward(input)

    def init_param(self, std=0.01):
        # 参数初始化
        self.params['weight'] = np.random.normal(loc=0.0, scale=std, size=(self.n_in, self.n_out))
        self.params['bias'] = np.zeros([1, self.n_out])

    def forward(self, input):
        # 前向传播计算
        self.input = input
        output = np.matmul(input, self.params['weight']) + self.params['bias']
        return output

    def backward(self, preGrad):
        # 反向传播计算
        self.d_params['weight'] = np.dot(self.input.T, preGrad)
        self.d_params['bias'] = np.sum(preGrad, axis=0)
        postGrad = np.dot(preGrad, self.params['weight'].T)
        return postGrad

    def update_param(self, lr, momentum):
        # SGD + momentum
        # 对全连接层参数利用参数进行更新(梯度下降)
        if self.pre is None:
            self.pre = {'weight': np.zeros_like(self.params['weight']), 'bias': np.zeros_like(self.params['bias'])}
        for key in self.params.keys():
            self.pre[key] = momentum * self.pre[key] - lr * self.d_params[key]
            self.params[key] += self.pre[key]

    def load_param(self, weight, bias):
        # 参数加载
        self.params['weight'] = weight
        self.params['bias'] = bias

    def save_param(self):
        # 参数保存
        return self.params['weight'], self.params['bias']


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


class MLP(object):
    """多层感知机
    """

    def __init__(self, train_data, test_data, batch_size=30, input_size=64, hidden1=256, hidden2=128, out_size=10,
                 lr=0.01, momentum=0.9, max_epoch=30):
        """
        :param train_data: 训练集
        :param test_data: 测试集
        :param batch_size: mini-batch 每次训练的样本数
        :param input_size: 输入维度
        :param hidden1: 第一层隐藏神经元数
        :param hidden2: 第二层隐藏神经元数
        :param out_size: 输出维度
        :param lr: 学习率
        :param momentum: 动量超参数
        :param max_epoch: 学习中所有训练数据均被使用过一次时的更新次数总数
        """
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_size = out_size
        self.lr = lr
        self.momentum = momentum
        self.max_epoch = max_epoch
        self.train_data = train_data
        self.test_data = test_data
        # 建立三层神经网络结构
        self.layer1 = Linear(self.input_size, self.hidden1)
        self.relu1 = Relu()
        self.layer2 = Linear(self.hidden1, self.hidden2)
        self.relu2 = Relu()
        self.layer3 = Linear(self.hidden2, self.out_size)
        self.softmax = SoftmaxWithLoss()
        self.layer_list = [self.layer1, self.layer2, self.layer3]

    def shuffle_data(self):
        # 打乱数据
        np.random.shuffle(self.train_data)

    def init_model(self):
        # 初始化多层感知机的全连接层
        for layer in self.layer_list:
            layer.init_param()

    def forward(self, input):
        # 神经网络的前向传播
        h1 = self.layer1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.layer2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.layer3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh3 = self.layer3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.layer2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        self.layer1.backward(dh1)

    def update_param(self, lr, momentum):
        # 更新每一层的参数
        for layer in self.layer_list:
            layer.update_param(lr, momentum)

    def save_param(self):
        # 保存每一层的参数
        params = {'w1': (self.layer1.save_param())[0],
                  'b1': (self.layer1.save_param())[1],
                  'w2': (self.layer2.save_param())[0],
                  'b2': (self.layer2.save_param())[1],
                  'w3': (self.layer3.save_param())[0],
                  'b3': (self.layer3.save_param())[1]}
        np.save('mnist.npy', params)

    def load_param(self):
        # 加载每一层的参数
        params = np.load('mnist.npy', allow_pickle=True).item()
        self.layer1.load_param(params['w1'], params['b1'])
        self.layer2.load_param(params['w2'], params['b2'])
        self.layer3.load_param(params['w3'], params['b3'])

    def train(self):
        # mini-batch训练模型
        loss = 0.0
        loss_list = []
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update_param(self.lr, self.momentum)
            loss_list.append(loss)
            print('Epoch %d, loss: %.6f' % (idx_epoch, loss))
        return loss_list

    def evaluate(self):
        # 评价模型准确度
        pred_results = np.zeros_like(self.test_data[:, -1])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)


def main():
    """调用神经网络模型完成mnist数据集上多分类任务

    :return: None
    """
    max_epoch = 100
    train_data, test_data = MNIST.load_data()
    mlp = MLP(train_data, test_data, max_epoch=max_epoch)
    mlp.init_model()
    loss_list = mlp.train()
    # mlp.save_param()
    # mlp.load_param()
    mlp.evaluate()
    plt.plot(list(range(max_epoch)), loss_list)
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    main()
