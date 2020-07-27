#  @author Duke Chain
#  @File:weight.py
#  @createTime 2020/07/26 20:11:26

import numpy as np


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
        return weightMatrix
    else:
        print("判断矩阵未通过一致性检验，ci的值为：", ci)
        return -1


AHPgetWeight(5, [0.5, 4, 3, 3, 7, 5, 5, 0.5, 1 / 3, 1])
