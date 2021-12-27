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
        """读取和预处理 MNIST 中训练数据和测试数据的图像和标签

        :return: 训练集与测试集
        """
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
        self.input_shape = None
        self.params = {}
        self.d_params = {}
        self.pre = None

    def __call__(self, input):
        return self.forward(input)

    def init_param(self, std=0.01):
        """系数初始化

        :param std:分布的标准差
        """
        self.params['weight'] = np.random.normal(loc=0.0, scale=std, size=(self.n_in, self.n_out))
        self.params['bias'] = np.zeros([1, self.n_out])

    def forward(self, input):
        """全连接层前向传播计算

        :param input: 输入数据数组
        :return: XW + b
        """
        # 更改张量形状
        self.input_shape = input.shape
        self.input = input.reshape(input.shape[0], -1)
        output = np.matmul(self.input, self.params['weight']) + self.params['bias']
        return output

    def backward(self, preGrad):
        """全连接层反向传播计算

        :param preGrad: 反向传播前一层的偏导数
        :return: loss对X的偏导数
        """
        self.d_params['weight'] = np.dot(self.input.T, preGrad)
        self.d_params['bias'] = np.sum(preGrad, axis=0)
        postGrad = np.dot(preGrad, self.params['weight'].T)
        # 还原张量形状
        postGrad = postGrad.reshape(*self.input_shape)
        return postGrad

    def update_param(self, lr, momentum):
        """对全连接层参数进行更新(使用SGD + momentum)

        :param lr: 学习率
        :param momentum: 动量超参数
        """
        if self.pre is None:
            self.pre = {'weight': np.zeros_like(self.params['weight']), 'bias': np.zeros_like(self.params['bias'])}
        for key in self.params.keys():
            self.pre[key] = momentum * self.pre[key] - lr * self.d_params[key]
            self.params[key] += self.pre[key]

    def load_param(self, weight, bias):
        """使该全连接层的权重矩阵和偏置向量更改为输入的值

        :param weight: 权重矩阵
        :param bias: 偏置向量
        """
        self.params['weight'] = weight
        self.params['bias'] = bias

    def save_param(self):
        """返回该全连接层的权重矩阵和偏置向量

        :return: weight,bias
        """
        return self.params['weight'], self.params['bias']


class Relu(object):
    """Relu 激活函数
    """

    def __init__(self):
        self.input = None

    def forward(self, input):
        """Relu 前向传播的计算

        :param input: 输入数据数组
        :return: maximum(0, input)
        """
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, preGrad):
        """Relu 反向传播的计算

        :param preGrad: 反向传播前一层的偏导数
        :return: 反向传播该层的偏导数
        """
        postGrad = preGrad
        postGrad[self.input < 0] = 0
        return postGrad


class Sigmoid(object):
    """sigmoid 激活函数
    """

    def __init__(self):
        self.output = None

    def forward(self, input):
        """Sigmoid 前向传播的计算

        :param input: 输入数据数组
        :return: 1 / (1 + exp(-input))
        """
        output = 1 / (1 + np.exp(-input))
        self.output = output
        return output

    def backward(self, preGrad):
        """Sigmoid 反向传播的计算

        :param preGrad: 反向传播前一层的偏导数
        :return: 反向传播该层的偏导数
        """
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
        """拉大了输入值之间的差异并将其归一化为一个概率分布，便于进行多分类

        :param input: 前向传播神经网络的计算结果
        :return: 概率分布数组
        """
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        """交叉熵误差损失函数，衡量预测值与真实值之间的差异程度

        :param label: 标签数组
        :return: 交叉熵损失值
        """
        self.batch_size = self.prob.shape[0]
        # onehot化
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label.astype(int)] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        """Softmax 反向传播

        :return: 反向传播该层的偏导数
        """
        postGrad = (self.prob - self.label_onehot) / self.batch_size
        return postGrad


class MLP(object):
    """多层感知机
    """

    def __init__(self, train_data, test_data, batch_size=32, input_size=64, hidden1=256, hidden2=128, out_size=10,
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
        :param max_epoch: 学习中所有训练数据均被使用过一次的更新次数总数
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
        self.all_list = [self.layer1, self.relu1, self.layer2, self.relu2, self.layer3, self.softmax]

    def shuffle_data(self):
        """打乱训练的数据集
        """
        np.random.shuffle(self.train_data)

    def init_model(self):
        """初始化多层感知机的全连接层参数(weight,bias)
        """
        for layer in self.layer_list:
            layer.init_param()

    def forward(self, input):
        """进行神经网络的前向传播

        :param input: mini-batch数据集图像数组
        :return: 神经网络预测概率数组
        """
        prob = input
        for layer in self.all_list:
            prob = layer.forward(prob)
        return prob

    def backward(self):
        """进行神经网络的反向传播，得到每一层的偏导值
        """
        dh = self.softmax.backward()
        for layer in self.all_list[-2::-1]:
            dh = layer.backward(dh)

    def update_param(self, lr, momentum):
        """更新网络中每一层全连接层的参数(weight,bias)

        :param lr: 学习率
        :param momentum: 动量超参数
        """
        for layer in self.layer_list:
            layer.update_param(lr, momentum)

    def save_param(self, address='mnist.npy'):
        """保存每一层的参数(weight,bias)
        """
        params = {}
        for i in range(len(self.layer_list)):
            w, b = self.layer_list[i].save_param()
            params['w{}'.format(i + 1)] = w
            params['b{}'.format(i + 1)] = b
        np.save(address, params)

    def load_param(self, address='mnist.npy'):
        """读取保存的参数(weight,bias)，并分别加载到每一层
        """
        params = np.load(address, allow_pickle=True).item()
        for i in range(len(self.layer_list)):
            self.layer_list[i].load_param(params['w{}'.format(i + 1)], params['b{}'.format(i + 1)])

    def train(self):
        """使用mini-batch学习训练神经网络模型

        :return: 每轮训练的损失值
        """
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
        """使用测试集数据运行训练后的神经网络模型，评价模型的准确度
        """
        prob = self.forward(self.test_data[:, :-1])
        pred_labels = np.argmax(prob, axis=1)
        accuracy = np.mean(pred_labels == self.test_data[:, -1])
        print('Accuracy in the test set: %f' % accuracy)


def main():
    """调用神经网络模型完成 mnist 数据集上多分类任务
    """
    max_epoch = 30
    train_data, test_data = MNIST.load_data()
    mlp = MLP(train_data, test_data, max_epoch=max_epoch)
    mlp.init_model()
    # mlp.load_param()
    loss_list = mlp.train()
    # mlp.save_param()
    mlp.evaluate()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(range(1, max_epoch + 1)), loss_list)
    plt.title("模型训练过程loss值变化图")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    main()
