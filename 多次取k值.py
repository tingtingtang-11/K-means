import matplotlib.pyplot as plt
import numpy as np


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


def cost_fun(note, k, m):
    cost_count = np.zeros((k,2))
    for i in range(m):
        # note[i][0] # 距离
        # note[i][1] # =j 所属簇中心j
        # cost_count[j][0] # j类距离总和
        # cost_count[j][1]  # j类个数
        cost_count[int(note[i][1])][0] += note[i][0]  # j类距离按类增加
        cost_count[int(note[i][1])][1] +=1  # j类个数
    cost = np.zeros((k,1))
    for i in range(k):
        cost[i] = cost_count[i][0]/cost_count[i][1]

    print(np.sum(cost,axis=1)/cost.shape[0])
    return cost_count


def kmean(datasets, k, dis_count):
    m = datasets.shape[0]
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
    print("初始化聚类中心中心后:" + str(np.sum(note[:,0])))
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
        # draw(datasets, note, center)     # 每次更新聚类中心与分簇都输出一个图像

    # cost_fun(note, k, m)
    dis_count.append(np.sum(note[:, 0])/m)
    print("第"+str(k-1)+"次聚类："+str(np.sum(note[:, 0])))
    print("第" + str(k - 1) + "次聚类：" + str(np.sum(note[:, 0])/m))
    draw(datasets,note,center,k)

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
    for k in range(2,11):
        note, center_index,dis_count = kmean(a, k, dis_count)
        print(dis_count)
    plt.plot([k for k in range (2,len(dis_count)+2)], dis_count)
    plt.show()



