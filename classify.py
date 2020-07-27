#  @author Duke Chain
#  @File:classify.py
#  @createTime 2020/07/27 20:34:27
import numpy as np


def classify(timeseries, weightmatrix):
    """
    时间序列节点聚类

    先对多维时间序列降维（权重提取），再通过k-means算法对降维后时间节点进行聚类，进而寻找异常点

    Args:
        timeseries(m*n的矩阵):传入初始时间序列
        weightmatrix(n*1的矩阵):传入权重矩阵

    Returns:
    """

    # 对时间序列进行降维（时间序列和权重矩阵点乘）
    series = np.dot(timeseries, weightmatrix)
    print("降维后的时间序列为：", series)

