#  @author Duke Chain
#  @File:auto_test.py
#  @createTime 2020/07/26 22:32:26
from sklearn.datasets import make_blobs
from classify import *
from weight import *

# x, y = make_blobs(n_samples=500,
#                   n_features=2,
#                   centers=4,
#                   random_state=1)
#
# kmeans_cluster(6, x)
#
# # weightMatrix = AHPgetWeight(5, [0.5, 4, 3, 3, 7, 5, 5, 0.5, 1 / 3, 1])
# # timeSeries = [[1, 32, 3, 14, 45],
# #               [6, 47, 18, 19, 10],
# #               [11, 2, 13, 14, 65],
# #               [16, 7, 18, 9, 20]]
# # data = preprocessing(timeSeries, weightMatrix)
# # kmeans_cluster(1, data)
# result=prediction("result.pkl", [[0, 0]])
# print(result)
