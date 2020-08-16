#  @author Duke Chain
#  @File:preprocess.py
#  @createTime 2020/08/10 14:32:10


import numpy as np


def generate_particle(timenum, dimension):
    """
    产生粒子

    使用高斯分布随机产生粒子

    Args:
        timenum:时间节点的数量
        dimension:每个时间节点元素的个数

    Returns:
        matrix:按照高斯分布的粒子矩阵
    """

    # 产生粒子矩阵
    matrix = np.random.randn(timenum, dimension)

    # 可视化粒子矩阵
    print(matrix)

    return matrix


def expected_value_slope(matrix):
    """
    计算预期值

    通过计算单个属性斜率来得出预期值

    Args:
        matrix:传入最少两行矩阵（需要经过numpy array方法处理）

    Returns:
        result:预期值
    """

    # 判断矩阵是否符合标准
    assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录元素个数
    colNum = matrix.shape[1]

    # 初始化result矩阵
    result = np.zeros([1, matrix.shape[1]])

    for col in range(colNum):
        # 求斜率
        k = matrix[-1, col] - matrix[-2, col]
        # 将预测值加入result矩阵
        result[0, col] = matrix[-1, col] + k

    return result

