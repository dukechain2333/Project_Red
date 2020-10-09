#  @author Duke Chain
#  @File:Main_1st.py
#  @createTime 2020/10/09 18:26:09

from second_attempt.particleFilter import *
from second_attempt.preprocess import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tushare as ts

def mainprocess(data):
    """
    主处理模块

    主处理模块实现了各个方法的整合

    Args:
        data:传入数据

    Returns:
        resultMatrix:预测的矩阵
    """

    # 记录数据信息
    rowNum = data.shape[0]
    colNum = data.shape[1]

    # 数据标准化
    scaler = preprocessing.MinMaxScaler()
    testParticles = scaler.fit_transform(data)

    timeNum_pre = np.zeros([rowNum + 1, 1])
    for i in range(rowNum + 1):
        timeNum_pre[i] = i

    timeNum = np.zeros([rowNum, 1])
    for i in range(rowNum):
        timeNum[i] = i

    # 生成结果矩阵
    resultMatrix = np.zeros([rowNum + 1, 1])

    # 生成预测矩阵
    expectMatrix = expected_value_ma(testParticles)

    # 套用方法
    for row in range(rowNum):
        expectValue = expectMatrix[row].reshape(1, colNum)
        particles = generate_particle(1000, 1) + expectValue
        weight = weight_calculate(particles, expectValue)
        result = russia_roulette(weight, particles)
        resultMatrix[row] = result

    # 预测点
    # expectValue = expectMatrix[-1].reshape(1, colNum)
    # particles = generate_particle(1000, 1) + expectValue
    # weight = weight_calculate(particles, expectValue)
    # result = russia_roulette(weight, particles)
    # resultMatrix[rowNum] = result

    # 结果矩阵标准化
    resultMatrix = scaler.fit_transform(resultMatrix)

    # 向左平移2个单位进行滞后修正
    resultMatrix_pro = np.zeros([rowNum, 1])
    for i in range(rowNum - 2):
        resultMatrix_pro[i] = resultMatrix[i + 2]
    resultMatrix_pro[-1] = resultMatrix[-1]
    resultMatrix_pro[-2] = resultMatrix[-1]

    # 绘制图像
    plt.plot(timeNum, testParticles)
    plt.plot(timeNum, resultMatrix_pro, color='red')
    plt.title('RESULT OF MA')
    plt.legend(['real', 'predict'])
    plt.show()
    print(resultMatrix_pro)

    return resultMatrix_pro


