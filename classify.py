#  @author Duke Chain
#  @File:classify.py
#  @createTime 2020/07/27 20:34:27
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


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
    print("降维后的时间序列为：\n", series)
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

    return collections


def kmeans_cluster(cluster, collections):
    """
    对处理后的序列聚类

    对通过preprocessing()方法处理后的时间序列进行k-means聚类

    Args:
        cluster(int):传入聚类簇的数量
        collections(m*2的矩阵)：传入一个经过preprocessing()处理的m*2矩阵

    Returns:
        输出聚类后的信息及图像
    """

    # 创建聚类器并聚类
    estimator = KMeans(n_clusters=cluster)
    result = estimator.fit(collections)

    # 保存模型
    joblib.dump(result, "result.pkl")

    # 统计聚类情况
    y_pred = MiniBatchKMeans(n_clusters=cluster, batch_size=89, random_state=9).fit_predict(collections)
    lable_collect = np.zeros([1, cluster])
    for i in y_pred:
        lable_collect[0, i] += 1

    print("共有%d个样本" % len(y_pred))
    print("样本标签如下：\n", y_pred)
    for i in range(cluster):
        print("标签%d的个数为：%d" % (i, lable_collect[0, i]))

    # 输出最有可能成为异常点的标签（样本数量最少的簇）
    min_element = lable_collect[0, 0]
    min_label = 0
    for i in range(cluster):
        if lable_collect[0, i] < min_element:
            min_element = lable_collect[0, i]
            min_label = i

    print("最有可能是异常标签的是：%d\n标签内共有样本：%d" % (min_label, min_element))

    print("------------------------------")

    # 可视化聚类结果
    predict_labels = result.labels_
    centers = result.cluster_centers_
    plt.scatter(collections[:, 0], collections[:, 1], c=predict_labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=80, marker='x')
    plt.xlabel("Index")
    plt.ylabel("Latitude")
    plt.title("The result of clustering")
    plt.show()


def prediction(model, data):
    """
    对新的时间节点进行预测

    通过载入kmeans_cluster中训练好的模型，对新的时间节点进行预测，判断其是否为异常点

    Args:
        model(训练好的模型):传入一个在kmeans_cluster中训练好的模型
        data(一个1*2的矩阵):传入一个需要预测的节点

    Returns:
        返回预测的标签
    """

    # 导入训练好的模型
    predictor = joblib.load(model)
    result = predictor.predict(data)

    return result
