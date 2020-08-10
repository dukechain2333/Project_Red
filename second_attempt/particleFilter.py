#  @author Duke Chain
#  @File:particleFilter.py
#  @createTime 2020/08/03 13:09:03

import numpy as np
from second_attempt.preprocess import expected_value
from math import sqrt, pi, exp


def weight_calculate(particleMatrix, originMatrix):
    """
    权重计算

    假设矩阵按照正态分布，则采用概率密度函数（P.D.F）求出点的权重，并进行归一化

    Args:
        particleMatrix:传入按照正态分布的随机粒子矩阵
        originMatrix:传入需要计算的矩阵

    Returns：
        weightMatrix:归一化后的权重矩阵
    """

    # 求出期望值
    expectedMatrix = expected_value(originMatrix)

    # 统计正态分布矩阵信息
    colNum = particleMatrix.shape[1]
    rowNum = particleMatrix.shape[0]

    # 初始化权重矩阵
    weightMatrixtem = np.zeros(particleMatrix.shape)
    weightMatrix = np.zeros(particleMatrix.shape)

    # 使用概率密度函数求出权重矩阵
    for row in range(rowNum):
        for col in range(colNum):
            weightMatrixtem[row, col] = (1 / sqrt(2 * pi)) * exp(
                (particleMatrix[row, col] - expectedMatrix[0, col] ** 2) / (-2))

    # 归一化权重矩阵
    weightSum = np.zeros([1, colNum])

    for i in range(rowNum):
        weightSum += weightMatrixtem[i]

    for row in range(rowNum):
        weightMatrix[row] = weightMatrixtem[row] / weightSum

    return weightMatrix
