import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pearsonar(X, y):
    pearson = []
    for col in X.columns.values:
        pearson.append(abs(pearsonr(X[col].values, y)[0]))
    pearsonr_X = pd.DataFrame({'col': X.columns, 'corr_value': pearson})
    pearsonr_X = pearsonr_X.sort_values(by='corr_value', ascending=False)
    return pearsonr_X

def feature_pearsonar(x_data):
    # 利用皮尔逊相关系数计算特征之间的线性关系
    c = list(combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2))
    p = []
    for i in range(0, len(c)):
        p.append(abs(pearsonr(x_data[:, c[i][0]], x_data[:, c[i][1]])[0]))
    pearsonr_ = pd.DataFrame({'col': c, 'corr_value': p})
    pearsonr_ = pearsonr_.sort_values(by='corr_value', ascending=False)
    print(pearsonr_)

    return None


def getData(url, columns):
    data = pd.read_csv(url, header=None, names=columns)
    print(data.head())
    # 数据信息，说明没有缺失值，不需要进行缺失值处理
    print(data.info())
    # 每个类别各有多少样本量
    print(data['0category'].value_counts())

    return data


def data_processing(data):
    # 样本的真实类别
    y_data = data['0category']
    # 样本的特征值，13个
    x_data = data.drop('0category', axis=1)
    print(x_data.info())

    # 标准化，对原始数据进行变换把数据变换到均值为0，标准差为1的数据
    sd = StandardScaler()
    x_data = sd.fit_transform(x_data)

    return x_data, y_data


if __name__ == "__main__":
    columns = ['0category', '1Alcohol', '2Malic acid ', '3Ash', '4Alcalinity of ash',
               '5Magnesium', '6Total phenols', '7Flavanoid',
               '8Nonflavanoid phenols', '9Proanthocyanins ', '10Color intensity ', '11Hue ',
               '12OD280/OD315 of diluted wines', '13Proline ']
    data = getData("wine.csv", columns)

    x_data, y_data = data_processing(data)

    feature_pearsonar(x_data)

    pca = PCA(n_components=0.95)
    x_data = pca.fit_transform(x_data)
    print(x_data.shape)

    selector = VarianceThreshold(threshold=0.16)
    x_data = selector.fit_transform(x_data)
    print(x_data.shape)

    # params = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # x_data = x_data[:,params]
    # print(x_data.shape)

    km = KMeans(n_clusters=3)
    km.fit(x_data)
    # 每个样本所属的类
    predict_pre = km.labels_
    print("===========================================")
    print("聚类结果：")
    print(predict_pre)

    # 兰德系数，衡量的是两个数据分布的吻合程度
    print("调整兰德系数是：" + str(metrics.adjusted_rand_score(y_data,predict_pre)))
    # V-measure
    print("同质性：" + str(metrics.homogeneity_score(y_data, predict_pre)))
    print("完整性：" + str(metrics.completeness_score(y_data, predict_pre)))
    print("两者的调和平均V-measure：" + str(metrics.v_measure_score(y_data, predict_pre)))
    # 轮廓系数
    print("轮廓系数：" + str(metrics.silhouette_score(x_data, predict_pre)))

    # 建立四个颜色的列表
    color = ['orange', 'green', 'blue']
    # 遍历列表，把预测结果标记成对应的颜色
    colr1 = [color[i] for i in predict_pre]
    plt.scatter(x_data[:, 1], x_data[:, 2], color=colr1)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


