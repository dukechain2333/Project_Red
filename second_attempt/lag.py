#  @author Duke Chain
#  @File:lag.py
#  @createTime 2020/08/19 22:14:19

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 看起来是随机的

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


# 自相关

num_points = 100
data = []
for x in range(num_points):
    y1 = x * 0.1 + 0.3 + np.random.normal(0.0, 0.5)
    data.append([y1])
data = np.array(data)


def lagplot(data):
    """
    绘制lag图像
    
    按列绘制lag图像以辅助选择合适的状态方程模型
    
    Args:
        data:传入数据
        
    Returns:
        lag图像
    
    """

    # 统计传入数据信息
    rowNum = data.shape[0]
    colNum = data.shape[1]

    # 对数据进行标准化
    scaler = preprocessing.MinMaxScaler()
    proData = scaler.fit_transform(data)
    proData = np.vsplit(proData, rowNum)

    # 对数据分列处理
    for col in range(colNum):

        # 准备绘制图像阶段
        mat_i = np.zeros([rowNum - 1, 1])
        for i in range(rowNum - 1):
            mat_i[i] = proData[i + 1]

        mat_im = np.zeros([rowNum - 1, 1])
        for im in range(rowNum - 1):
            mat_im[im] = proData[im]

        plt.figure()
        plt.scatter(mat_im, mat_i)
        plt.title('Column No.%d' % col)
        plt.xlabel('i-1')
        plt.ylabel('i')
        plt.show()


lagplot(data)
