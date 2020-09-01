#  @author Duke Chain
#  @File:preprocess.py
#  @createTime 2020/08/10 14:32:10


import numpy as np
from sklearn import linear_model
import joblib
import matplotlib.pyplot as plt


def generate_particle(rownum, dimension):
    """
    产生粒子

    使用高斯分布随机产生粒子

    Args:
        rownum:生成数据点的数量
        dimension:每个时间节点元素的个数

    Returns:
        matrix:按照高斯分布的粒子矩阵
    """

    # 产生粒子矩阵
    matrix = np.random.randn(rownum, dimension)

    # 可视化粒子矩阵
    # print(matrix)

    return matrix


def expected_value_slope(row_one, row_two):
    """
    计算预期值

    通过计算单个属性斜率来得出预期值

    Args:
        row_one:传入第一行矩阵
        row_two:传入第二行矩阵

    Returns:
        result:期望值
    """

    # 判断矩阵是否符合标准
    # assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录元素个数
    colNum = row_one.shape[1]

    # 初始化result矩阵
    result = np.zeros([1, colNum])

    for col in range(colNum):
        # 求斜率
        k = row_two[-1, col] - row_one[-1, col]
        # 将预测值加入result矩阵
        result[0, col] = row_two[-1, col] + k

    return result


def expected_value_linear(matrix):
    """
    计算期望值（AR）

    通过自回归模型（此处采用线性自回归模型）来实现期望值的预测
    注意：使用此方法需要调用训练好的线性模型

    Args:
        matrix:传入最少两行矩阵（需要经过numpy array方法处理）

    Returns:
        输出一个线性模型
    """

    # 判断矩阵是否符合标准
    assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录元素个数
    rowNum = matrix.shape[0]

    # 初始化行矩阵
    rowMatrix = np.zeros([rowNum, 1])
    for row in range(rowNum):
        rowMatrix[row, 0] = row

    print('正在训练线性模型')

    # 通过sklearn的LinearRegression()进行训练
    model = linear_model.LinearRegression()
    model.fit(rowMatrix, matrix)

    # 保存模型
    joblib.dump(model, 'linear_model.pkl')
    print('线性模型已保存')



def expected_value_ma(matrix):
    """
    计算期望值（MA）

    通过滑动平均模型来实现期望值的预测
    注意：在例子滤波中使用此方法需要进行向左平移2单位的滞后性修正

    Args:
        matrix:传入最少两行矩阵（需要经过numpy array方法处理）

    Returns:
        result:期望值矩阵
    """

    # 判断矩阵是否符合标准
    assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录矩阵属性
    rowNum = matrix.shape[0]
    colNum = matrix.shape[1]

    # 设置缓存区取样个数并初始化缓存区
    cacheLen = 10

    # 初始化结果矩阵
    result = np.zeros([rowNum, colNum])

    # 按矩阵的列进行划分运算
    for col in range(colNum):
        # 进行基于正态分布的滑动平均滤波
        for row in range(rowNum):
            if row >= cacheLen:
                cache = matrix[row - cacheLen:row, col]
                meanValue = cache.mean()
                stdValue = cache.std()
                for element in range(10):
                    if abs(cache[element] - meanValue) > 1.96 * stdValue:
                        cache[element] = meanValue
                ma = cache.mean()
                result[row, col] = ma
            else:
                cache = matrix[0:row, col]
                if len(cache) != 0:
                    meanValue = cache.mean()
                    stdValue = cache.std()
                    for element in range(len(cache)):
                        if abs(cache[element] - meanValue) > 1.96 * stdValue:
                            cache[element] = meanValue
                    ma = cache.mean()
                    result[row, col] = ma
                else:
                    result[row, col] = matrix[row, col]

    # 绘图
    # time = np.zeros([rowNum, 1])
    # for row in range(rowNum):
    #     time[row] = row
    # for col in range(colNum):
    #     plt.plot(time, result[:, col], color='red')
    #     plt.plot(time, matrix[:, col])
    #     plt.legend(['MA', 'REAL'])
    #     plt.title('Column No.%d' % col)
    #     plt.show()

    # 返回预测结果
    return result
