#  @author Duke Chain
#  @File:MainComprenhensive.py
#  @createTime 2020/10/14 13:58:14

from second_attempt.particleFilter import *
from second_attempt.preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import threading
import time
import pandas as pd
from fastdtw import fastdtw


class thread_ar(threading.Thread):
    def __init__(self, data):
        threading.Thread.__init__(self)
        # self.func = func
        self.data = data
        self.result = self.ar(self.data)

    def ar(self, data):
        # 数据标准化
        scaler = preprocessing.MinMaxScaler()

        rowNum = data.shape[0]
        colNum = data.shape[1]

        testParticles_ar = scaler.fit_transform(data)

        # 训练线性模型
        expected_value_linear(testParticles_ar)
        expectModel = joblib.load('linear_model.pkl')

        # 方法套用
        resultMatrix_ar = np.zeros([rowNum, 1])
        for row in range(rowNum):
            expectValue = expectModel.predict([[row]])
            particles = generate_particle(1000, 1) + expectValue
            weight = weight_calculate(particles, expectValue)
            result = russia_roulette(weight, particles)
            resultMatrix_ar[row] = result

        # # 预测点
        # expectValue = expectModel.predict([[rowNum]])
        # particles = generate_particle(1000, 1) + expectValue
        # weight = weight_calculate(particles, expectValue)
        # result = russia_roulette(weight, particles)
        # resultMatrix_ar[rowNum] = result

        resultMatrix_ar = scaler.fit_transform(resultMatrix_ar)

        return resultMatrix_ar

    def get_result(self):
        return self.result


class thread_ma(threading.Thread):
    def __init__(self, data):
        threading.Thread.__init__(self)
        # self.func = func
        self.data = data
        self.result = self.ma(self.data)

    def ma(self, data):
        # 数据标准化
        scaler = preprocessing.MinMaxScaler()

        rowNum = data.shape[0]
        colNum = data.shape[1]

        testParticles_ma = scaler.fit_transform(data)

        # 生成结果矩阵
        resultMatrix = np.zeros([rowNum, 1])

        # 生成预测矩阵
        expectMatrix = expected_value_ma(testParticles_ma)

        # 套用方法
        for row in range(rowNum):
            expectValue = expectMatrix[row].reshape(1, colNum)
            particles = generate_particle(1000, 1) + expectValue
            weight = weight_calculate(particles, expectValue)
            result = russia_roulette(weight, particles)
            resultMatrix[row] = result

        # 预测点
        # expectValue = expectMatrix[-1].reshape(1, colNum)
        # particles = generate_particle(1000, 1) + expectValue
        # weight = weight_calculate(particles, expectValue)
        # result = russia_roulette(weight, particles)
        # resultMatrix[rowNum] = result

        # 结果矩阵标准化
        resultMatrix = scaler.fit_transform(resultMatrix)

        # 向左平移2个单位进行滞后修正
        resultMatrix_ma = np.zeros([rowNum, 1])
        for row in range(rowNum - 2):
            resultMatrix_ma[row] = resultMatrix[row + 2]
        resultMatrix_ma[-1] = resultMatrix[-1]
        resultMatrix_ma[-2] = resultMatrix[-1]

        return resultMatrix_ma

    def get_result(self):
        return self.result


data = np.array([[3052446],
                 [4375598],
                 [1353297],
                 [2851196],
                 [4057208],
                 [5546890],
                 [1632174],
                 [3458323],
                 [5537990],
                 [7660146],
                 [2173374],
                 [4722226],
                 [6908835],
                 [9389332],
                 [2683745],
                 [5574482],
                 [8183720],
                 [10847321],
                 [2536566],
                 [5077783],
                 [7627113],
                 [11204251],
                 [2431087],
                 [5179156],
                 [8060342],
                 [11403678],
                 [2944620],
                 [5924985],
                 [9162457],
                 [12637868],
                 [3139691],
                 [7104397],
                 [11072568],
                 [15296763],
                 [4240685],
                 [8409622],
                 [13520532],
                 [18112951],
                 [3036735]])

# data = np.zeros([100, 1])
# for i in range(100):
#     data[i] = np.sin(i) + np.random.normal(0, 0.1)

if __name__ == '__main__':

    df = pd.read_csv('D:\\Project_Red\\data_daily\\002047_SZ.csv')
    count = []
    for num in range(2, 101):
        data = df['open'][150:150 + num]
        data = np.array(data)
        data = data.reshape(-1, 1)

        correctCollective = []

        for times in range(100):
            print(times)
            # ar = thread_ar(data)
            ma = thread_ma(data)
            # ar.start()
            ma.start()

            # resultMatrix_ar = ar.get_result()
            resultMatrix_ma = ma.get_result()
            scaler = preprocessing.MinMaxScaler()
            testParticles = scaler.fit_transform(data)

            # 趋势正确性检测
            correct = 0
            for i in range(resultMatrix_ma.shape[0] - 1):
                realTend = testParticles[i + 1] - testParticles[i]
                predictTend = resultMatrix_ma[i + 1] - resultMatrix_ma[i]
                if realTend * predictTend >= 0:
                    correct += 1
            correctRate = correct / (resultMatrix_ma.shape[0] - 1)
            correctCollective.append(correctRate)
            # print("正确率：", correctRate)

            # similar_ma, path_ma = fastdtw(resultMatrix_ma, testParticles)
            # similar_ar, path_ar = fastdtw(resultMatrix_ar, testParticles)

            # if similar_ma > similar_ar:
            #     print("AR模型更加合适")
            # elif similar_ma < similar_ar:
            #     print("MA模型更加合适")
            # else:
            #     print("AR和MA均合适")

            # rowNum = data.shape[0]
            # colNum = data.shape[1]
            #
            # timeNum_pre = np.zeros([rowNum + 1, 1])
            # for i in range(rowNum + 1):
            #     timeNum_pre[i] = i
            #
            # timeNum = np.zeros([rowNum, 1])
            # for i in range(rowNum):
            #     timeNum[i] = i

            # plt.clf()
            # plt.plot(timeNum, testParticles)
            # plt.plot(timeNum, resultMatrix_ma, color='red')
            # plt.plot(timeNum, resultMatrix_ar, color='green')
            # plt.title('RESULT OF MA & AR')
            # plt.title('RESULT OF MA')
            # plt.legend(['real', 'MA', 'AR'])
            # plt.savefig('D:\\Project_Red\\test_pic\\test1\\pic' + str(times) + '.png')
            # plt.show()
            # print("testParticels:")
            # print(testParticles)
            # print("predict:")
            # print(resultMatrix_ma)

        # print(correctCollective)
        # print('平均正确率为：', sum(correctCollective) / len(correctCollective))

        count.append(sum(correctCollective) / len(correctCollective))

    plt.plot([x for x in range(1, len(count) + 1)], count)
    plt.title("Latitude of Accuracy")
    plt.savefig("D:\\Project_Red\\picAccuracy2.png")
    plt.show()
