import matplotlib.pyplot as plt
import numpy as np

# 增加初始化的次数，选择代价函数值最小的情况作为初始的聚类中心

def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)


def get_random_center(datasets,k):
    center_index = np.random.randint(0,200,k)
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


def kmean(datasets, k, dis_count):
    m = datasets.shape[0]
    good_note = np.empty((m,2))
    good_init_center_index = np.empty((k,2))
    min_total_dist = 10000
    for num in range(10):
        note = np.zeros((m,2))
        center_index,center = get_random_center(datasets,k)
        for i in range(m):
            minDist = 100000
            center_number = 100
            for j in range(k):
                dis = distEclud(center[j],datasets[i])
                if minDist > dis:
                    minDist = dis
                    center_number = j
            note[i][0] = minDist
            note[i][1] = center_number
        print("第" + str(num + 1) + "次初始化聚类中心中心后:" + str(np.sum(note[:,0])))
        draw(datasets, note, center,k)

        centerChanged = True
        while centerChanged:
            centerChanged = False
            center = new_center(datasets, note, k)
            for i in range(m):
                minDist = 100000
                center_number = 100
                for j in range(k):
                    dis = distEclud(center[j],datasets[i])
                    if minDist > dis:
                        minDist = dis
                        center_number = j
                note[i][0] = minDist
                if note[i][1] != center_number:
                    centerChanged = True
                    note[i][1] = center_number

        average_dis = np.sum(note[:, 0])/m
        if average_dis < min_total_dist:
            min_total_dist = average_dis
            good_note = note
            good_init_center_index = center_index
        dis_count.append(average_dis)
        print("第" + str(num + 1) + "次初始化代价函数值：" + str(average_dis))
        draw(datasets,note,center,k)
    print("最好的效果时，初始化的聚类中心是：")
    print(good_init_center_index)

    return note,center_index,dis_count


def draw(datasets,note,center,k):
    colors = ['black', 'red', 'blue', 'yellow','orange','purple','green','brown','pink','grey','olive']
    for i in range(datasets.shape[0]):
        for j in range(k):
            if note[i][1] == float(j):
                plt.scatter(datasets[i][0], datasets[i][1], color=colors[j])
    plt.scatter(center[:, 0], center[:, 1], color='green', marker='+',s=60)
    plt.show()

if __name__ == "__main__":
    np.seterr(divide='ignore',invalid='ignore')

    a = np.random.randint(0,50,size=[200,2])
    dis_count = []
    note, center_index,dis_count = kmean(a, 4, dis_count)

    print(dis_count)
    plt.plot([k for k in range (1,len(dis_count)+1)], dis_count)
    plt.show()