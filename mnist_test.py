#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Neural_Networks
@File    ：mnist_test.py
@Author  ：ypf
@Date    ：2021/12/22 19:02
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

digits = datasets.load_digits()
train_images, test_images, train_labels, test_labels \
    = train_test_split(digits.data, digits.target, test_size=0.2)
some_digit = train_images[0]
some_digit_image = some_digit.reshape(8, 8)
print(some_digit_image)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')
plt.show()
print(train_labels[0])


