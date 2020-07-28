#  @author Duke Chain
#  @File:classify.py
#  @createTime 2020/07/27 20:34:27
import numpy as np
from weight import AHPgetWeight


def preprocessing(timeseries, weightmatrix):
    """
    时间序列节点预处理

    对多维时间序列进行降维处理（多维时间序列点乘权重矩阵）

    Args:
        timeseries(m*n的矩阵):传入初始时间序列
        weightmatrix(n*1的矩阵):传入权重矩阵

    Returns:
        collections(m*2的数组)
    """

    # 对时间序列进行降维（时间序列和权重矩阵点乘）
    series = np.dot(timeseries, weightmatrix)
    print("降维后的时间序列为：", series)
    print("------------------------------")

    # 对数据进行预处理及重构
    m = len(series)
    collections = np.zeros([m - 1, 2])

    for i in range(m - 1):
        latitude = series[i + 1] - series[i]
        collections[i][0] = series[i]
        collections[i][1] = latitude

    print("完成预处理流程后的数据为：\n", collections)
    print("------------------------------")


weightMatrix = AHPgetWeight(5, [0.5, 4, 3, 3, 7, 5, 5, 0.5, 1 / 3, 1])
timeSeries = [[1, 32, 3, 14, 45],
              [6, 47, 18, 19, 10],
              [11, 2, 13, 14, 65],
              [16, 7, 18, 9, 20]]
preprocessing(timeSeries, weightMatrix)
