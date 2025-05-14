#coding=gbk
import random
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt

import warnings
# �����漶������Ϊ ERROR����ζ��ֻ����ʾ ERROR ����ľ���

class hypergraph(object):
    def __init__(self):
        self.m0 = int(input("�������ʼ�ڵ����m0:"))
        self.m1 = int(input("������ÿ�����ӵ��½ڵ���m1:")) # ÿ�������½ڵ����
        self.m2 = int(input("������ѡȡ�ľɽڵ���m2:")) # ÿ������ѡ��ɽڵ����
        self.timestep = int(input("������ʱ�䲽t:"))
        self.m = int(input("������ÿ�����ӵĳ�����m:"))  # ÿ�������³��ߵ�����(���ܴ��ڳ�ʼ�ڵ����4����ȡ2,3,4)
        self.n = int(input("���������������ھӽ���n:"))  # �ھӽڵ�Ľ���
        self.nodesNum = self.m0 + self.m1 * self.timestep # �����ܽڵ����
        self.lastNode = 0  # ��¼��ǰ���һ���ڵ�ı��
        self.lastEdge = 1  # ��¼��ǰ���һ�����ߵı��
        self.incidenceMatrix = np.zeros((self.nodesNum, 1), int)  # ��������(�����о��ж��ٸ��ڵ㣬�����о��ж���������)
        self.scenes = {}
        self.xdata = []
        self.ydata = []
        self.dataSave = "ModelSave/"
    def funRun(self):
        self.modelCreate()
        self.exportData()
    def modelCreate(self):
        print("��������ģ��")
        # ������ʼ����
        eTuple = ()  # �����еĽڵ���Ԫ��洢
        for i in range(self.m0):
            # ���ڵ���뵽������
            eTuple += ('v' + str(i + 1),)
            # ���¹�������
            self.incidenceMatrix[i][0] = 1
        self.lastNode = self.m0
        self.scenes['E1'] = eTuple
        self.H = hnx.Hypergraph(self.scenes)
        # ��������
        newCol = [[0] for _ in range(len(self.incidenceMatrix))]
        while self.lastNode < self.nodesNum:
            print("\r���ȣ�" + str(self.lastNode + 1) + "/" + str(self.nodesNum), end='')
            current_m = 0  # ������¼��ǰ�Ѿ����ɶ������µĳ���
            # ѡ���³��ߵľɽڵ�
            node1 = random.randint(1, self.lastNode) #���ѡ��Ľڵ���
            neighbor_list = []
            neighbor_list.append("v" + str(node1))
            neighbor_list.extend(list(self.H.neighbors("v" + str(node1))))  # ��ѡ�ڵ��һ���ھ�
            neighbor_list = list(set(neighbor_list))  # ���ھӽڵ��ŵ�ȥ��
            neighbor_list.sort() # ���ھӽڵ��ŵ�����
            neighbor_listn = neighbor_list.copy()  # ��ѡ�ڵ��n���ھ�
            # print(neighbor_list)
            for i in range(1, self.n):
                for j in neighbor_list:
                    neighbor_listn.extend(list(self.H.neighbors(j)))
                neighbor_listn = list(set(neighbor_listn))  # ���ھӽڵ��ŵ�ȥ��
                neighbor_list = neighbor_listn.copy()

            while current_m < self.m:  # ����m������
                self.lastEdge += 1
                # �������ڵ��ȷ������ɵ�һ���³��ߣ������¹�������
                self.incidenceMatrix = np.c_[self.incidenceMatrix, newCol]
                eTuple = ()
                for i in range(1, self.m1 + 1):
                    eTuple += ('v' + str(self.lastNode + i),)
                    self.incidenceMatrix[self.lastNode + i - 1][self.lastEdge - 1] = 1
                current_m += 1
                selectNode_had = []
                selectNode = self.local_selection(neighbor_listn)  # local_selection�������ȸ��ʷ��ؾɽڵ��б�
                while selectNode in selectNode_had:   # ���ɲ�ͬ��m������
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
    # ���ո���ѡ��ڵ�
    def local_selection(self, neighbor_list):
        # �ȼ���ѡȡ�ĸ���
        pNode = []  # �洢���ڵ�Ķ�ռ�ܶ����ı���
        sumD = 0  # ��ͼ�нڵ���ܳ�����
        for i in neighbor_list:
            sumD += np.sum(np.array(self.incidenceMatrix[int(i[1:])-1]))
        for i in neighbor_list:
            pNode.append((int(i[1:]),round(np.sum(np.array(self.incidenceMatrix[int(i[1:])-1])) / sumD, 6)))  # ���ʱ���4λС��

        # �Ը������鴦��ʹ֮���䵽��0��1������
        pNodeSection = []  # �洢�����ڵ�����䣬����ʹ��Ԫ��洢
        left = right = 0  # ��������
        for p in pNode:
            left = right
            right += p[1]
            pNodeSection.append((p[0],(left, right)))

        # �����ظ�ѡ�㣬�ظ�������ѡ��
        selectNode = []  # �洢ѡȡ�Ľڵ���
        state = [True] * len(neighbor_list)  # �洢ѡȡ״̬
        existNode = self.m2  # ѡ��ľɽڵ���
        while existNode:
            for t in range(len(neighbor_list)):
                if state[t] and pNodeSection[t][1][0] <= round(random.random(), 6) < pNodeSection[t][1][1]:
                    selectNode.append(pNodeSection[t][0])
                    existNode -= 1
                    state[t] = False  # ���Ϊ��ѡ��
                if existNode == 0:
                    break
        return selectNode

    def degreeDistribute(self):
        print('���ڼ��㳬�ȷֲ�')
        N = self.nodesNum
        self.H = hnx.Hypergraph(self.scenes)
        # ���㳬��hk
        hk = hnx.degree_dist(self.H)
        # Pk �������㳬�ȷֲ�
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
        # plt.figure("�������ɷֲ�P(k)��k������ϵͼ", figsize=(10, 8))
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
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)
        ax.legend(labels=[r"emulational", "theoretical"], ncol=1, fontsize=20)
        #ax.legend(labels=[r"$p=0.5$", "theoretical"], ncol=1, fontsize=20)
        plt.savefig("img2.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    def exportData(self):
        print("�洢��...")
        file_path = self.dataSave +"n"+str(self.nodesNum)+"m1"+str(self.m1)+"m2"+str(self.m2)+"m"+str(self.m) + "_"+str(self.n)+'version.txt'
        with open(file_path, 'w') as file0:
            for i in range(self.nodesNum):
                print("\r���ȣ�" + str(i + 1) + '/' + str(self.nodesNum), end='')
                row_data = ' '.join(str(self.incidenceMatrix[i][j]) for j in range(self.lastEdge))
                print(row_data, file=file0)
        print("\n�洢��ɣ�" + file_path)

if __name__ == '__main__':
    H = hypergraph()
    H.funRun()
    H.degreeDistribute()
    H.theoretical()
    H.drawshow()