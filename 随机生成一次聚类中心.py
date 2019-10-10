import matplotlib.pyplot as plt
import numpy as np


def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)


def get_random_center(datasets,k):
    center_index = np.random.randint(0,50,k)
    center = np.zeros((k,2))
    for j in range(k):
        center[j] = datasets[center_index[j]]
    print(center)
    print(center_index)
    return center_index,center


def new_center(datasets,note,k):
    """
    更新聚类中心
    """
    # 样本数量
    m = datasets.shape[0]
    # 新的聚类中心
    new_center = np.zeros((k,3))

    for i in range(m):
        new_center[int(note[i][1])][0] += datasets[i][0]
        new_center[int(note[i][1])][1] += datasets[i][1]
        new_center[int(note[i][1])][2] += 1

    center = np.zeros((k,2))
    for i in range(k):
        center[i] = new_center[i,0:2]/new_center[i][2]

    return center


def kmean(datasets, k):
    note = np.zeros((50,2))
    center_index,center = get_random_center(datasets,k)
    for i in range(datasets.shape[0]):
        minDist = 100000
        center_number = 1000
        for j in range(k):
            dis = distEclud(center[j],datasets[i])
            if minDist > dis:
                minDist = dis
                center_number = j
        note[i][0] = minDist
        note[i][1] = center_number
    return note,center

def newkmean(datasets,k,note):
    center = new_center(datasets, note, k)
    new_note = np.zeros((50,2))
    for i in range(datasets.shape[0]):
        minDist = 100000
        center_number = 1000
        for j in range(k):
            dis = distEclud(datasets[i],center[j])
            if minDist > dis:
                minDist = dis
                center_number = j
        new_note[i][0] = minDist
        new_note[i][1] = center_number
    return new_note,center


def draw(datasets,note,center,k):
    colors = ['black', 'red', 'blue', 'yellow','orange','purple','green','brown','pink','grey','olive']
    for i in range(datasets.shape[0]):
        for j in range(k):
            if note[i][1] == float(j):
                plt.scatter(datasets[i][0], datasets[i][1], color=colors[j])
    plt.scatter(center[:, 0], center[:, 1], color='green', marker='+',s=100)
    plt.show()

if __name__ == "__main__":
    a = np.random.randint(0,100,size=[50,2])
    print(a)
    note,center = kmean(a, 3)

    draw(a, note, center, 3)

    for times in range(3):
        note,center = newkmean(a,3,note)
        draw(a, note, center, 3)
