#  @author Duke Chain
#  @File:main_AR.py
#  @createTime 2020/08/19 16:14:19

from second_attempt.particleFilter import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 生成状态数据
# data = np.array([[3052446],
#                  [4375598],
#                  [1353297],
#                  [2851196],
#                  [4057208],
#                  [5546890],
#                  [1632174],
#                  [3458323],
#                  [5537990],
#                  [7660146],
#                  [2173374],
#                  [4722226],
#                  [6908835],
#                  [9389332],
#                  [2683745],
#                  [5574482],
#                  [8183720],
#                  [10847321],
#                  [2536566],
#                  [5077783],
#                  [7627113],
#                  [11204251],
#                  [2431087],
#                  [5179156],
#                  [8060342],
#                  [11403678],
#                  [2944620],
#                  [5924985],
#                  [9162457],
#                  [12637868],
#                  [3139691],
#                  [7104397],
#                  [11072568],
#                  [15296763],
#                  [4240685],
#                  [8409622],
#                  [13520532],
#                  [18112951],
#                  [3036735]])

data = np.array([[1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6],
                 [7],
                 [8],
                 [9],
                 [10],
                 [11],
                 [12],
                 [13],
                 [14],
                 [15],
                 [16],
                 [17],
])

# num_points = 100
# data = []
# for i in range(num_points):
#     y1 = i * 0.1 + 0.3 + np.random.normal(0.0, 0.5)
#     data.append([y1])
# data = np.array(data)


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

    # 方法套用
    for i in range(rowNum - 1):
        expectValue = expected_value_linear(testParticles, i)
        # print(expectValue)
        particles = generate_particle(1000, 1) + expectValue
        weight = weight_calculate(particles, expectValue)
        result = russia_roulette(weight, particles)
        # print(result)
        resultMatrix[i + 1] = result

    resultMatrix = scaler.fit_transform(resultMatrix)
    resultMatrix[0] = testParticles[0]

    # 绘制图像
    plt.plot(timeNum, testParticles)
    plt.plot(timeNum_pre, resultMatrix, color='red')
    plt.legend(['real', 'predict'])
    plt.show()

    return resultMatrix


if __name__ == '__main__':
    mainprocess(data)
