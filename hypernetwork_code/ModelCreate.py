#coding=gbk
import random
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt

import warnings
# 将警告级别设置为 ERROR，意味着只会显示 ERROR 级别的警告

class hypergraph(object):
    def __init__(self):
        self.m0 = int(input("请输入初始节点个数m0:"))
        self.m1 = int(input("请输入每次增加的新节点数m1:")) # 每次增加新节点个数
        self.m2 = int(input("请输入选取的旧节点数m2:")) # 每条超边选择旧节点个数
        self.timestep = int(input("请输入时间步t:"))
        self.m = int(input("请输入每次增加的超边数m:"))  # 每次生成新超边的条数(不能大于初始节点个数4个，取2,3,4)
        self.n = int(input("请输入局域世界的邻居阶数n:"))  # 邻居节点的阶数
        self.nodesNum = self.m0 + self.m1 * self.timestep # 生成总节点个数
        self.lastNode = 0  # 记录当前最后一个节点的编号
        self.lastEdge = 1  # 记录当前最后一条超边的编号
        self.incidenceMatrix = np.zeros((self.nodesNum, 1), int)  # 关联矩阵(多少行就有多少个节点，多少列就有多少条超边)
        self.scenes = {}
        self.xdata = []
        self.ydata = []
        self.dataSave = "ModelSave/"
    def funRun(self):
        self.modelCreate()
        self.exportData()
    def modelCreate(self):
        print("正在生成模型")
        # 构建初始网络
        eTuple = ()  # 超边中的节点用元组存储
        for i in range(self.m0):
            # 将节点加入到超边中
            eTuple += ('v' + str(i + 1),)
            # 更新关联矩阵
            self.incidenceMatrix[i][0] = 1
        self.lastNode = self.m0
        self.scenes['E1'] = eTuple
        self.H = hnx.Hypergraph(self.scenes)
        # 网络增长
        newCol = [[0] for _ in range(len(self.incidenceMatrix))]
        while self.lastNode < self.nodesNum:
            print("\r进度：" + str(self.lastNode + 1) + "/" + str(self.nodesNum), end='')
            current_m = 0  # 用来记录当前已经生成多少条新的超边
            # 选择新超边的旧节点
            node1 = random.randint(1, self.lastNode) #随机选择的节点编号
            neighbor_list = []
            neighbor_list.append("v" + str(node1))
            neighbor_list.extend(list(self.H.neighbors("v" + str(node1))))  # 所选节点的一阶邻居
            neighbor_list = list(set(neighbor_list))  # 对邻居节点编号的去重
            neighbor_list.sort() # 对邻居节点编号的排序
            neighbor_listn = neighbor_list.copy()  # 所选节点的n阶邻居
            # print(neighbor_list)
            for i in range(1, self.n):
                for j in neighbor_list:
                    neighbor_listn.extend(list(self.H.neighbors(j)))
                neighbor_listn = list(set(neighbor_listn))  # 对邻居节点编号的去重
                neighbor_list = neighbor_listn.copy()

            while current_m < self.m:  # 生成m条超边
                self.lastEdge += 1
                # 将新增节点先放入生成的一个新超边，并更新关联矩阵
                self.incidenceMatrix = np.c_[self.incidenceMatrix, newCol]
                eTuple = ()
                for i in range(1, self.m1 + 1):
                    eTuple += ('v' + str(self.lastNode + i),)
                    self.incidenceMatrix[self.lastNode + i - 1][self.lastEdge - 1] = 1
                current_m += 1
                selectNode_had = []
                selectNode = self.local_selection(neighbor_listn)  # local_selection局域优先概率返回旧节点列表
                while selectNode in selectNode_had:   # 生成不同的m条超边
                    selectNode = self.local_selection(neighbor_listn)
                for i in range(self.m2):
                    eTuple += ('v' + str(selectNode[i]),)
                    self.incidenceMatrix[selectNode[i] - 1][self.lastEdge - 1] = 1
                self.scenes['E'+str(self.lastEdge)] = eTuple
                self.H = hnx.Hypergraph(self.scenes)
                selectNode_had.append(selectNode)
                # print(self.scenes)
            warnings.filterwarnings("error")
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.lastNode += self.m1
        #hnx.draw(self.H)
        # print(list(self.H.nodes))
        #plt.show()
        print(self.scenes)
        # for i in range(len(self.incidenceMatrix)):
        #     print(self.incidenceMatrix[i])
        print(self.incidenceMatrix)
    # 按照概率选择节点
    def local_selection(self, neighbor_list):
        # 先计算选取的概率
        pNode = []  # 存储各节点的度占总度数的比例
        sumD = 0  # 超图中节点的总超度数
        for i in neighbor_list:
            sumD += np.sum(np.array(self.incidenceMatrix[int(i[1:])-1]))
        for i in neighbor_list:
            pNode.append((int(i[1:]),round(np.sum(np.array(self.incidenceMatrix[int(i[1:])-1])) / sumD, 6)))  # 概率保留4位小数

        # 对概率数组处理，使之分配到【0，1】区间
        pNodeSection = []  # 存储各个节点的区间，区间使用元组存储
        left = right = 0  # 左右区间
        for p in pNode:
            left = right
            right += p[1]
            pNodeSection.append((p[0],(left, right)))

        # 不能重复选点，重复则重新选择
        selectNode = []  # 存储选取的节点编号
        state = [True] * len(neighbor_list)  # 存储选取状态
        existNode = self.m2  # 选择的旧节点数
        while existNode:
            for t in range(len(neighbor_list)):
                if state[t] and pNodeSection[t][1][0] <= round(random.random(), 6) < pNodeSection[t][1][1]:
                    selectNode.append(pNodeSection[t][0])
                    existNode -= 1
                    state[t] = False  # 标记为已选择
                if existNode == 0:
                    break
        return selectNode

    def degreeDistribute(self):
        print('正在计算超度分布')
        N = self.nodesNum
        self.H = hnx.Hypergraph(self.scenes)
        # 计算超度hk
        hk = hnx.degree_dist(self.H)
        # Pk 用来计算超度分布
        Pk = np.zeros(max(hk), float)
        for i in range(N):
            Pk[hk[i] - 1] += 1 / N
        x = [i for i in range(1, max(hk) + 1)]
        self.xdata.append(x)
        self.ydata.append(Pk)

    def theoretical(self):
        Pk = np.zeros(35, float)
        for k in range(1, 35):
            Pk[k] = (1/self.m)*(self.m1/self.m2+1)*((self.m/k)**(self.m1/self.m2+2))

        x = [i for i in range(1, 35)]
        self.xdata.append(x)
        self.ydata.append(Pk[1:])

    def drawshow(self):
        # plt.figure("超度幂律分布P(k)与k对数关系图", figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        # plt.scatter(self.xdata[0], self.ydata[0], marker='o', facecolors='none', edgecolors='blue')
        # plt.loglog(self.xdata[1], self.ydata[1], '-', lw=4, color='k')
        # plt.scatter(self.xdata[0], self.ydata[0], marker='o', facecolors='none', edgecolors='#ff7f0e')
        # plt.loglog(self.xdata[1], self.ydata[1], '-', lw=1, color='#1f77b4')
        plt.scatter(self.xdata[0], self.ydata[0], marker='o', facecolors='none', edgecolors='#e377c2')
        plt.loglog(self.xdata[1], self.ydata[1], '-', lw=1, color='k')
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel("k", fontsize=25)
        plt.ylabel("P(k)", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)
        ax.legend(labels=[r"emulational", "theoretical"], ncol=1, fontsize=20)
        #ax.legend(labels=[r"$p=0.5$", "theoretical"], ncol=1, fontsize=20)
        plt.savefig("img2.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    def exportData(self):
        print("存储中...")
        file_path = self.dataSave +"n"+str(self.nodesNum)+"m1"+str(self.m1)+"m2"+str(self.m2)+"m"+str(self.m) + "_"+str(self.n)+'version.txt'
        with open(file_path, 'w') as file0:
            for i in range(self.nodesNum):
                print("\r进度：" + str(i + 1) + '/' + str(self.nodesNum), end='')
                row_data = ' '.join(str(self.incidenceMatrix[i][j]) for j in range(self.lastEdge))
                print(row_data, file=file0)
        print("\n存储完成：" + file_path)

if __name__ == '__main__':
    H = hypergraph()
    H.funRun()
    H.degreeDistribute()
    H.theoretical()
    H.drawshow()
