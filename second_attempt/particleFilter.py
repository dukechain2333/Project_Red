#  @author Duke Chain
#  @File:particleFilter.py
#  @createTime 2020/08/03 13:09:03

import numpy as np
from second_attempt.preprocess import expected_value
from math import sqrt, pi, exp
import random


def weight_calculate(particlematrix, originmatrix):
    """
    权重计算

    假设矩阵按照正态分布，则采用概率密度函数（P.D.F）求出点的权重，并进行归一化

    Args:
        particlematrix:传入按照正态分布的随机粒子矩阵
        originmatrix:传入需要计算的矩阵

    Returns：
        weightMatrix:归一化后的权重矩阵
    """

    # 求出期望值
    expectedMatrix = expected_value(originmatrix)

    # 统计正态分布矩阵信息
    colNum = particlematrix.shape[1]
    rowNum = particlematrix.shape[0]

    # 初始化权重矩阵
    weightMatrixtem = np.zeros(particlematrix.shape)
    weightMatrix = np.zeros(particlematrix.shape)

    # 使用概率密度函数求出权重矩阵
    for row in range(rowNum):
        for col in range(colNum):
            weightMatrixtem[row, col] = (1 / sqrt(2 * pi)) * exp(
                (particlematrix[row, col] - expectedMatrix[0, col] ** 2) / (-2))

    # 归一化权重矩阵
    weightSum = np.zeros([1, colNum])

    for i in range(rowNum):
        weightSum += weightMatrixtem[i]

    for row in range(rowNum):
        weightMatrix[row] = weightMatrixtem[row] / weightSum

    return weightMatrix


def russia_roulette(weightmatrix, particlematrix):
    """
    重抽样环节

    以点的权重为标准进行重抽样

    Args:
        weightmatrix:传入权重矩阵
        particlematrix:传入按照正态分布的随机粒子矩阵

    Returns:
        result:返回进行重抽样后矩阵的均值，作为最终预测结果
    """

    # 统计正态分布矩阵信息
    colNum = particlematrix.shape[1]
    rowNum = particlematrix.shape[0]

    # 初始化重抽样矩阵
    selection = np.zeros([rowNum, colNum])

    # 求出抽样矩阵
    for col in range(colNum):
        # 计算单列权重总和
        colTotal = sum(weightmatrix[:, col])

        for row in range(rowNum):
            # 抽取随机权重
            selectionRate = random.uniform(0, colTotal)
            # 求出权重所在区间
            for secondRow in range(rowNum):
                if selectionRate < weightmatrix[secondRow, col]:
                    break
                selectionRate -= weightmatrix[secondRow, col]
            # 将结果映射到重抽样矩阵中
            selection[row, col] = particlematrix[secondRow, col]

    print("重抽样矩阵为：\n", selection)

    # 统计重抽样矩阵信息
    rowSel = selection.shape[0]
    colSel = selection.shape[1]

    # 初始化结果矩阵
    result = np.zeros([1, colSel])

    # 按列计算重抽样矩阵均值
    for col in range(colSel):
        total = sum(selection[:, col])
        colMean = total / rowSel
        result[0, col] = colMean

    # 返回结果矩阵
    return result
