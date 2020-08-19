#  @author Duke Chain
#  @File:main_AR.py
#  @createTime 2020/08/19 16:14:19

from second_attempt.particleFilter import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 生成状态数据
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

# num_points = 100
# data = []
# for i in range(num_points):
#     y1 = i * 0.1 + 0.3 + np.random.normal(0.0, 0.5)
#     data.append([y1])
# data=np.array(data)

scaler = preprocessing.MinMaxScaler()
testParticles = scaler.fit_transform(data)

timeNum_pre = np.zeros([40, 1])
for i in range(40):
    timeNum_pre[i] = i

timeNum = np.zeros([39, 1])
for i in range(39):
    timeNum[i] = i

resultMatrix = np.zeros([40, 1])

for i in range(39):
    expectValue = expected_value_linear(testParticles, i)
    # print(expectValue)
    particles = generate_particle(1000, 1) + expectValue
    weight = weight_calculate(particles, expectValue)
    result = russia_roulette(weight, particles)
    # print(result)
    resultMatrix[i + 1] = result

resultMatrix = scaler.fit_transform(resultMatrix)
resultMatrix[0] = testParticles[0]

plt.plot(timeNum, testParticles)
plt.plot(timeNum_pre, resultMatrix, color='red')
plt.show()
