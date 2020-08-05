#  @author Duke Chain
#  @File:particleFilter.py
#  @createTime 2020/08/03 13:09:03

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

