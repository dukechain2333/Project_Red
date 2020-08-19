#  @author Duke Chain
#  @File:preprocess.py
#  @createTime 2020/08/10 14:32:10


import numpy as np
from sklearn import linear_model
import joblib
import os.path


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


def expected_value_slope(rowOne, rowTwo):
    """
    计算预期值

    通过计算单个属性斜率来得出预期值

    Args:
        rowOne:传入第一行矩阵
        rowTwo:传入第二行矩阵

    Returns:
        result:期望值
    """

    # 判断矩阵是否符合标准
    # assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录元素个数
    colNum = rowOne.shape[1]

    # 初始化result矩阵
    result = np.zeros([1, colNum])

    for col in range(colNum):
        # 求斜率
        k = rowTwo[-1, col] - rowOne[-1, col]
        # 将预测值加入result矩阵
        result[0, col] = rowTwo[-1, col] + k

    return result


def expected_value_linear(matrix, row):
    """
        计算预期值（AR）

        通过自回归模型（此处采用线性自回归模型）来实现期望值的预测

        Args:
            matrix:传入最少两行矩阵（需要经过numpy array方法处理）
            row:需要预测的节点

        Returns:
            result:期望值
        """

    # 判断矩阵是否符合标准
    assert (matrix.shape[0] >= 2), 'expected_value()方法出现错误：传入矩阵不符合要求'

    # 记录元素个数row
    rowNum = matrix.shape[0]

    # 检查是否已有训练模型
    if not os.path.isfile('linear_model.pkl'):

        # 初始化行矩阵
        rowMatrix = np.zeros([rowNum, 1])
        for row in range(rowNum):
            rowMatrix[row, 0] = row

        # 通过sklearn的LinearRegression()进行训练
        model = linear_model.LinearRegression()
        model.fit(rowMatrix, matrix)

        # 保存模型
        joblib.dump(model, 'linear_model.pkl')
        print('线性模型已保存')

    # 读取已保存的模型
    linearModel = joblib.load('linear_model.pkl')

    # 通过模型进行预测
    result = linearModel.predict([[row]])

    return result
