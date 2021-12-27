#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Neural_Networks
@File    ：CNN_test.py
@Author  ：ypf
@Date    ：2021/12/26 12:17
"""
import numpy as np


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


in_mat = np.array(
    [
        [
            [1, 2, 1], [1, 1, 1], [1, 2, 1]
        ],
        [
            [1, 1, 1], [1, 1, 1], [1, 1, 1]
        ],
        [
            [1, 1, 1], [1, 1, 1], [1, 1, 1]
        ]
    ]
)

ker = np.array(
    [
        [
            [1, 1], [1, 1]
        ],
        [
            [1, 1], [1, 1]
        ],
        [
            [1, 1], [1, 1]
        ]
    ]
)
print(conv2D(1, in_mat, ker))
