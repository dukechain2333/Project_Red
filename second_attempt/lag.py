#  @author Duke Chain
#  @File:lag.py
#  @createTime 2020/08/19 22:14:19

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 看起来是随机的

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

scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)
data = np.vsplit(data, 39)

mat_i = np.zeros([38, 1])
for i in range(38):
    mat_i[i] = data[i + 1]
print(mat_i)

mat_im = np.zeros([38, 1])
for i in range(38):
    mat_im[i] = data[i]
print(mat_im)

plt.figure()
plt.scatter(mat_im, mat_i)
plt.xlabel('i-1')
plt.ylabel('i')
plt.show()

# 自相关

# num_points = 100
# data = []
# for i in range(num_points):
#     y1 = i * 0.1 + 0.3 + np.random.normal(0.0, 0.5)
#     data.append([y1])
# data=np.array(data)
#
# scaler = preprocessing.MinMaxScaler()
# data = scaler.fit_transform(data)
# data=np.vsplit(data,100)
#
# mat_i=np.zeros([99,1])
# for i in range(99):
#     mat_i[i]=data[i+1]
# print(mat_i)
#
# mat_im=np.zeros([99,1])
# for i in range(99):
#     mat_im[i]=data[i]
# print(mat_im)
#
# plt.figure()
# plt.scatter(mat_im,mat_i)
# plt.xlabel('i-1')
# plt.ylabel('i')
# plt.show()
