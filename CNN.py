#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Neural_Networks
@File    ：CNN.py
@Author  ：易鹏飞
@Content ：numpy卷积神经网络完成mnist数据集上多分类任务
"""
import matplotlib.pyplot as plt
import numpy as np
from Neural_network import MNIST
from Neural_network import Relu
from Neural_network import Linear
from Neural_network import SoftmaxWithLoss
from Neural_network import MLP


def reshape_data(data):
    """将mnist的图片转化为四维数组shape = (N,C,W,H)

    :param data: 图像数据
    :return: 四维数组shape = (N,C,W,H)
    """
    N = data.shape[0]
    out = []
    for i in range(N):
        out_i = data[i].reshape(8, 8)
        out.append([out_i])
    return np.array(out)


class Im_col(object):
    @staticmethod
    def im2col(input, kernel_w, kernel_h, stride=1, pad=0):
        """

        :param input: shape = (N,C,W,H)
        :param kernel_w: 卷积核的宽
        :param kernel_h: 卷积核的高
        :param stride: 步长
        :param pad: 填充
        :return: col二维数组
        """

        N, C, W, H = input.shape
        out_w = (W + 2 * pad - kernel_w) // stride + 1
        out_h = (H + 2 * pad - kernel_h) // stride + 1

        img = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, kernel_w, kernel_h, out_w, out_h))

        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                col[:, :, x, y, :, :] = img[:, :, x:x_max:stride, y:y_max:stride]

        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * out_w * out_h, -1)
        return col

    @staticmethod
    def col2im(col, input_shape, kernel_w, kernel_h, stride=1, pad=0):
        """

        :param col: 二维数组
        :param input_shape: (N,C,W,H)
        :param kernel_w: 卷积核的宽
        :param kernel_h: 卷积核的高
        :param stride: 步长
        :param pad: 填充
        :return: 数组shape = (N,C,W,H)
        """

        N, C, W, H = input_shape
        out_w = (W + 2 * pad - kernel_h) // stride + 1
        out_h = (H + 2 * pad - kernel_w) // stride + 1
        col = col.reshape(N, out_w, out_h, C, kernel_w, kernel_h).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, W + 2 * pad + stride - 1, H + 2 * pad + stride - 1))
        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                img[:, :, x:x_max:stride, y:y_max:stride] += col[:, :, x, y, :, :]

        return img[:, :, pad:W + pad, pad:H + pad]


class Conv(object):
    """卷积层
    """

    def __init__(self, kernel_shape, stride=1, pad=0):
        """

        :param kernel_shape: (N,C,W,H)
        :param stride: 步长
        :param pad: 填充
        """
        self.kernel_shape = kernel_shape
        self.kernel = None
        self.bias = None
        self.stride = stride
        self.pad = pad
        self.input = None
        self.col = None
        self.col_kernel = None
        self.d_kernel = None
        self.d_bias = None

    def init_param(self, std=0.01):
        """系数初始化

        :param std: 标准化系数
        """
        self.kernel = std * np.random.randn(*self.kernel_shape)
        self.bias = np.zeros(self.kernel_shape[0])

    def forward(self, input):
        """卷积层前向传播

        :param input: 输入数据数组
        :return: 卷积值
        """
        self.input = input
        FN, C, FW, FH = self.kernel.shape
        N, C, W, H = self.input.shape
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        # col化input与kernel
        self.col = Im_col.im2col(self.input, FW, FH, self.stride, self.pad)
        self.col_kernel = self.kernel.reshape(FN, -1).T
        # 卷积计算
        out = np.dot(self.col, self.col_kernel) + self.bias
        out = out.reshape(N, out_w, out_h, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, d_out):
        """卷积层反向传播

        :param d_out: 反向传播前一层的偏导数
        :return: 反向传播该层偏导数
        """
        FN, C, FW, FH = self.kernel.shape
        d_out = d_out.transpose((0, 2, 3, 1)).reshape(-1, FN)
        # 求偏导(梯度)
        self.d_bias = np.sum(d_out, axis=0)
        self.d_kernel = np.dot(self.col.T, d_out)
        self.d_kernel = self.d_kernel.transpose((1, 0)).reshape(FN, C, FH, FW)
        # 卷积计算
        d_col = np.dot(d_out, self.col_kernel.T)
        d_input = Im_col.col2im(d_col, self.input.shape, FW, FH, self.stride, self.pad)
        return d_input

    def update_param(self, lr=0.01, *args):
        """更新参数kernel,bias(SGD方法)

        :param lr: 学习率
        """
        self.kernel -= lr * self.d_kernel
        self.bias -= lr * self.d_bias

    def load_param(self, kernel, bias):
        """使该卷积层的卷积核和偏置向量更改为输入的值

        :param kernel: 卷积核
        :param bias: 偏置向量
        """
        self.kernel = kernel
        self.bias = bias

    def save_param(self):
        """返回该卷积层的卷积核和偏置向量

        :return: kernel,bias
        """
        return self.kernel, self.bias


class Pooling(object):
    """池化层(缩小W,H方向上的空间)
    """

    def __init__(self, pool_w, pool_h, stride=1, pad=0):
        """

        :param pool_w: 目标区域宽
        :param pool_h: 目标区域高
        :param stride: 步长
        :param pad: 填充
        """
        self.pool_w = pool_w
        self.pool_h = pool_h
        self.stride = stride
        self.pad = pad
        self.input = None
        self.arg_max = None

    def forward(self, input):
        """池化层前向传播

        :param input: 输入数据数组
        :return: 缩小W,H方向上的空间后值
        """
        self.input = input
        N, C, W, H = self.input.shape
        out_w = int(1 + (W - self.pool_w) / self.stride)
        out_h = int(1 + (H - self.pool_h) / self.stride)
        # col化
        col = Im_col.im2col(self.input, self.pool_w, self.pool_h, self.stride, self.pad)
        col = col.reshape(-1, self.pool_w * self.pool_h)
        # 计算max
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_w, out_h, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, d_out):
        """池化层反向传播

        :param d_out: 反向传播前一层的偏导数
        :return: 反向传播该层的偏导数
        """
        d_out = d_out.transpose(0, 2, 3, 1)
        # 计算偏导(梯度)
        pool_size = self.pool_w * self.pool_h
        d_max = np.zeros((d_out.size, pool_size))
        d_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_out.flatten()
        d_max = d_max.reshape(d_out.shape + (pool_size,))
        d_col = d_max.reshape(d_max.shape[0] * d_max.shape[1] * d_max.shape[2], -1)
        d_input = Im_col.col2im(d_col, self.input.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return d_input


class CNN(MLP):
    """构建卷积神经网络
    """

    def __init__(self, train_data, test_data, batch_size=32, input_size=(1, 8, 8), conv_param=None, hidden1=128,
                 out_size=10, lr=0.01, momentum=0.9, max_epoch=30):
        """

        :param train_data: 训练集
        :param test_data: 测试集
        :param batch_size: mini-batch 每次训练的样本数
        :param input_size: 输入维度
        :param conv_param: 卷积层参数
        :param hidden1: 隐藏神经元数
        :param out_size: 输出维度
        :param lr: 学习率
        :param momentum: 动量超参数
        :param max_epoch: 学习中所有训练数据均被使用过一次的更新次数总数
        """
        if conv_param is None:
            conv_param = {'kernel_shape': (30, 1, 3, 3), 'stride': 1, 'pad': 0}
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.out_size = out_size
        self.lr = lr
        self.momentum = momentum
        self.max_epoch = max_epoch
        self.train_data = train_data
        self.test_data = test_data
        self.conv_param = conv_param
        conv_output_size = (input_size[1] - conv_param.get('kernel_shape')[2] + 2 * conv_param.get(
            'pad')) / conv_param.get('stride') + 1
        pool_output_size = int(conv_param.get('kernel_shape')[0] * (conv_output_size / 2) * (conv_output_size / 2))
        # 构建卷积神经网络
        self.conv1 = Conv(self.conv_param.get('kernel_shape'), self.conv_param.get('stride'),
                          self.conv_param.get('pad'))
        self.relu1 = Relu()
        self.pool1 = Pooling(pool_w=2, pool_h=2, stride=2)
        self.layer1 = Linear(pool_output_size, self.hidden1)
        self.relu2 = Relu()
        self.layer2 = Linear(self.hidden1, self.out_size)
        self.softmax = SoftmaxWithLoss()
        self.layer_list = [self.conv1, self.layer1, self.layer2]
        self.all_list = [self.conv1, self.relu1, self.pool1, self.layer1, self.relu2, self.layer2, self.softmax]

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
                batch_images = reshape_data(batch_images)
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
        prob = self.forward(reshape_data(self.test_data[:, :-1]))
        pred_labels = np.argmax(prob, axis=1)
        accuracy = np.mean(pred_labels == self.test_data[:, -1])
        print('Accuracy in the test set: %f' % accuracy)


def main():
    """用卷积神经网络模型完成 mnist 数据集上多分类任务
    """
    max_epoch = 100
    train_data, test_data = MNIST.load_data()
    cnn = CNN(train_data, test_data, max_epoch=max_epoch)
    cnn.init_model()
    # cnn.load_param('mnist_cnn.npy')
    loss = cnn.train()
    # cnn.save_param('mnist_cnn.npy')
    cnn.evaluate()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(range(1, max_epoch + 1)), loss)
    plt.title("模型训练过程loss值变化图")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    main()
