#  @author Duke Chain
#  @File:auto_test.py
#  @createTime 2020/07/26 22:32:26
from sklearn.datasets import make_blobs
from classify import *

x, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  random_state=1)

kmeans_cluster(3, x)
