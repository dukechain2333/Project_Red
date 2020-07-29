#  @author Duke Chain
#  @File:weight.py
#  @createTime 2020/07/26 20:11:26

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def AHPgetWeight(dimension, relation):
    """
    权重计算

    通过AHP（层次分析法）进行权重的计算
    权重的计算通过几何平均法实现

    Args:
        dimension(int):传入属性的个数作为矩阵的维度
        relation(数组):传入属性正向的相关度（属性之间不可与自身比较）
        
    Returns:
        返回一个dimension*1(weightMatrix)的矩阵(数组)作为权重矩阵

    """

    # 给判断矩阵赋初值
    judgeMatrix = np.zeros([dimension, dimension])
    n = 0  # relation元素的计数器
    while n <= dimension - 1:
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    judgeMatrix[i, j] = 1
                else:
                    if judgeMatrix[i, j] == 0:
                        judgeMatrix[i, j] = relation[n]
                        judgeMatrix[j, i] = 1 / relation[n]
                        n += 1
    print("判断矩阵为：\n", judgeMatrix)
    print("------------------------------")

    # 通过几何平均法进行权重计算
    weightMatrix = np.zeros(dimension)
    for i in range(dimension):
        up = 1  # 分子
        for j in range(dimension):
            up *= judgeMatrix[i, j]
        up = up ** (1 / dimension)

        down = 0  # 分母
        for x in range(dimension):
            temp = 1
            for x2 in range(dimension):
                temp *= judgeMatrix[x, x2]
            temp = temp ** (1 / dimension)
            down += temp

        weightMatrix[i] = up / down

    # weightMatrix转置
    weightMatrix = weightMatrix.reshape(dimension, 1)
    print("权重矩阵为：\n", weightMatrix)
    print("------------------------------")

    # 判断一致性比率是否达标
    # cr=ci/ri
    eigenValue = np.linalg.eig(judgeMatrix)
    maxEigenvalue = eigenValue[0][0]
    ci = (maxEigenvalue - dimension) / (dimension - 1)

    if dimension == 1:
        ri = 0.001
    elif dimension == 2:
        ri = 0.001
    elif dimension == 3:
        ri = 0.58
    elif dimension == 4:
        ri = 0.90
    elif dimension == 5:
        ri = 1.12
    elif dimension == 6:
        ri = 1.24
    elif dimension == 7:
        ri = 1.32
    elif dimension == 8:
        ri = 1.41
    elif dimension == 9:
        ri = 1.45
    elif dimension == 10:
        ri = 1.49
    else:
        ri = 1.51

    if ci / ri < 0.1:
        print("判断矩阵通过一致性检验，一致性比率为：", ci / ri)
        print("------------------------------")
        return weightMatrix
    else:
        print("判断矩阵未通过一致性检验，ci的值为：", ci)
        print("------------------------------")
        return -1


def dimensionShow(dimension):
    """
        维度运算图像显示

        显示在dimension维的矩阵中共有多少种权重可能性

        Args:
            dimension(int):传入属性的个数作为矩阵的维度

        Returns:
            显示可能性与dimension的关系

        """

    # y轴取值
    n = 3
    number = np.zeros([9, 1])
    while n <= 11:
        total = 0
        for i in range(1, n):
            total += i
        number[n - 3] = 18 ** total
        n += 1

    # 输出在dimension维度情况下的可能性
    print("在%s维度的情况下，共有%f种判断矩阵" % (dimension, number[dimension - 3]))
    print("------------------------------")

    # 将可能性组合进行min-max标准化处理
    number = np.array(number)
    presser = preprocessing.MinMaxScaler()
    number_pro = presser.fit_transform(number)

    # x轴范围
    epoch = range(3, 12)

    plt.xlabel("Dimension")
    plt.ylabel("Times")
    plt.title("dimension&times'relation")
    plt.plot(epoch, number_pro)
    plt.show()
