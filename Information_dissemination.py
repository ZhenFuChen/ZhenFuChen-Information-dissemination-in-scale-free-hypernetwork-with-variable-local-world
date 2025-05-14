#coding=gbk
import random
import numpy as np
import hypernetx as hnx
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
# �����漶������Ϊ ERROR����ζ��ֻ����ʾ ERROR ����ľ���
class InfoSpreading(object):
    def __init__(self):
        self.nodesNum = 0
        self.beta = 0
        self.gamma = 0
        self.timeStep = 50 # ʱ�䲽
        self.repeatNum = 100  # �ظ�ʵ�����(Ĭ��100)
        self.betaNum = 21
        self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
        self.imageData = np.zeros((self.timeStep, 20), float)  # ���ÿ��ʱ�䲽�Ħѵ��������
        self.stateData = []
        self.scenes = {}
        self.node_to_edge = {}
        self.imgSave = "../Save/ImgSave/"
        self.importPath = "ModelSave/"
        self.dataSave ="../Save/DataSave/ResultData/"

    # ��̬�¦ѵķ���
    def funRun1(self):
        # ����ģ��
        self.importModel(1000,3,3,3,2)
        for beta in range(self.betaNum):
            print("\r��="+str(beta/100),end="")
            self.sprSteadyState(beta/100, 0.3, 0, "avgHi")
        print(self.stateData)
        #self.dataExport()  # ���ݴ洢
        self.imgDraw()  # ��ͼ

    # ��ͬ�ڵ���
    def funRun2(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000,3,3,3,2)
        self.spr(0.1,0.1,0, "avgHi")
        self.importModel(3000,3,3,3,2)
        self.spr(0.1,0.1,1, "avgHi")
        self.importModel(5000,3,3,3,2)
        self.spr(0.1,0.1,2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw2()  # ��ͼ

    # ��ͬ�����ʦ�
    def funRun3(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.05,0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp_raw(0.2, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw3()  # ��ͼ

    # ��ͬ�ָ��ʦ�
    def funRun4(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.05, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.10, 1, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.20, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw4()  # ��ͼ

    # ��ͬ�ھӽ���n
    def funRun5(self):
        # ����ģ��
        # ���߱�������
        # self.importModel_raw(1000,3,3,3)
        # self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,3)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,3,3,3,4)
        self.spr(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw5()  # ��ͼ

    # ��ͬ��ʼ�����ڵ�
    def funRun6(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 0, "minHi")
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 2, "maxHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw6()  # ��ͼ

    # ��BA������Ա�
    def funRun7(self):
        # ����ģ��
        # ���߱�������
        self.importModel_raw(1000,3,3,3)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel_raw(1000,3,3,3)
        self.spr(0.1, 0.1, 1, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw7()  # ��ͼ

    # ��ͬm1
    def funRun8(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000,1,3,3,2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,5,3,3,2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # �·�����������ʵ�����߻�ͼʱʹ�ã��޸�beta
        # self.importModel(1000, 1, 3, 3)
        # self.spr(0.2, 0.1, 0, "avgHi")
        # self.importModel(1000, 3, 3, 3)
        # self.spr(0.2, 0.1, 1, "avgHi")
        # self.importModel(1000, 5, 3, 3)
        # self.spr(0.2, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw8()  # ��ͼ

    # ��ͬm2
    def funRun9(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000, 3, 1, 3, 2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000, 3, 3, 3, 2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3, 5, 3, 2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw9()  # ��ͼ

    # ��ͬm
    def funRun10(self):
        # ����ģ��
        # ���߱�������
        self.importModel(1000, 3, 3, 1, 2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000, 3, 3, 3, 2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3, 3, 5, 2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw10()  # ��ͼ

    # m1+m2��Ӱ��
    def funRun11(self):
        # ����ģ��
        # ���߱�������
        # self.importModel(1000, 1, 1, 3, 2)
        # self.spr_cp(0.1, 0.1, 0, "avgHi")
        # self.importModel(1000, 3, 3, 3, 2)
        # self.spr_cp(0.1, 0.1, 1, "avgHi")
        # self.importModel(1000, 5, 5, 3, 2)
        # self.spr_cp(0.1, 0.1, 2, "avgHi")

        self.importModel(1000, 1, 5, 3, 2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000, 3, 3, 3, 2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 5, 1, 3, 2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # ���ݴ洢
        self.imgDraw11()  # ��ͼ

    # ��ͬp,m2�仯��Ӱ��
    def funRun12(self):
        self.importPath = "ModelSave/"
        self.importModel(999, 3, 1,3, 2)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(999, 3, 3, 3, 2)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(999, 3, 1, 3, 8)
        self.spr(0.1, 0.1, 2, "avgHi")
        self.importModel(999, 3, 3, 3, 8)
        self.spr(0.1, 0.1, 3, "avgHi")

        self.imgDraw12()  # ��ͼ

    # ��ͬp,m1�仯��Ӱ��
    def funRun13(self):
        self.importPath = "ModelSave/"
        self.importModel(999, 1, 3, 3, 2)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(997, 3, 3, 3, 2)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(999, 1, 3, 3, 8)
        self.spr(0.1, 0.1, 2, "avgHi")
        self.importModel(997, 3, 3, 3, 8)
        self.spr(0.1, 0.1, 3, "avgHi")

        self.imgDraw13()  # ��ͼ

    # ��������
    def importModel(self, N,m1,m2,m,n):  # �ڵ���(1000,3000,5000),������,��ֵ(ʵ��ֵ��ʮ��),ģ��(edge,node)
        self.nodesNum = N
        self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
        self.scenes = {}
        print("����n" + str(N) +"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+"n"+str(n) + "ģ��")
        # ������ģ�͵���
        self.incidenceMatrix = np.genfromtxt(self.importPath+"n"+str(self.nodesNum)+"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+"_"+str(n)+"version.txt",delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # �����еĽڵ���Ԫ��洢
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
            self.scenes["E" + str(i + 1)] = eTuple
        edgesNum = len(self.incidenceMatrix[0]) #������ĳ�����
        for node, edge_list in enumerate(self.incidenceMatrix, start=1):
            edges = ()
            for i in range(edgesNum):
                if edge_list[i] != 0:
                    edges += ('E' + str(i + 1),)
            self.node_to_edge["v" + str(node)] = edges

    def importModel_raw(self, N,m1,m2,m):  # �ڵ���(1000,3000,5000),������,��ֵ(ʵ��ֵ��ʮ��),ģ��(edge,node)
        self.nodesNum = N
        self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
        self.scenes = {}
        print("����n" + str(N) +"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+ "ģ��")
        # ������ģ�͵���
        self.incidenceMatrix = np.genfromtxt(self.importPath+"n"+str(self.nodesNum)+"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+".txt",delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # �����еĽڵ���Ԫ��洢
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
                    # if len(eTuple) == 4:
                    #     break
            self.scenes["E" + str(i + 1)] = eTuple

    # ��������
    def importModel1(self, N, E, theta, m1,m2,m):  # �ڵ���(1000,3000,5000),������,��ֵ(ʵ��ֵ��ʮ��),ģ��(edge,node)
        self.nodesNum = N
        self.theta = theta
        self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
        self.scenes = {}
        print("���ڵ���n{}e{}theta{}m1{}m2{}m{}��ģ��".format(N,E,theta,m1,m2,m))
        # ������ģ�͵���
        self.incidenceMatrix = np.genfromtxt(
            self.importPath + "n" + str(N) + "e" + str(E) + "theta" + str(
                theta) + "m1"+str(m1)+"m2"+str(m2)+"m"+str(m) + "EdgeInc.txt", delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # �����еĽڵ���Ԫ��洢
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
                    # if len(eTuple) == 4:
                    #     break
            self.scenes["E" + str(i + 1)] = eTuple

    # ��Ϣ����
    def spr(self, beita, gama, n, select):
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        for repeat in range(self.repeatNum):
            print("\r�ظ�ʵ����ȣ�"+str(repeat+1)+"/"+str(self.repeatNum),end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

            edgesNum = len(self.incidenceMatrix[0])  # ������ĳ�����
            # ����t��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)
                # ɸѡ��I״̬�ڵ�ĳ���
                for e in range(edgesNum):
                    for node in self.scenes['E' + str(e + 1)]:
                        node_id = int(node[1:]) - 1
                        if self.state[node_id][1] == 1:  # �˳��ߵ�����I״̬�ڵ�
                            for i in self.scenes['E' + str(e + 1)]:
                                i_id = int(i[1:]) - 1
                                if self.state[i_id][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                                    if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                        self.IStateNodes.append(i_id + 1)  # I״̬�ڵ㵱����Ӵ˽ڵ���
                # �޸�����
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�
                # print("�ָ��ڵ㣺",recoverNodes)
                # print("��Ⱦ�ڵ㣺",self.IStateNodes)
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1
                for i in recoverNodes:
                    self.state[i - 1][1] = 0
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum
        print()

    def spr_cp(self, beita, gama, n, select):
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        for repeat in range(self.repeatNum):
            print("\r�ظ�ʵ����ȣ�" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

            # ���� t ��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []  # �ָ��ڵ��б�
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)

                for node in self.IStateNodes:
                    chosen_edge = random.choice(self.node_to_edge["v" + str(node)]) #ȡ��I�ڵ���������г���
                    # print(chosen_edge)
                    for chosen_node in self.scenes[chosen_edge]:
                        if self.state[int(chosen_node[1:]) - 1][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                            if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                self.IStateNodes.append(int(chosen_node[1:]))  # I״̬�ڵ㵱����Ӵ˽ڵ���

                #self.incidenceMatrix ��Ϊ�ڵ㣬��Ϊ����
                # �޸����ݣ�����״̬
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]  # ɾ���ָ��Ľڵ�
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�

                # ����״̬
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1  # ��I״̬�ڵ��״̬��ΪI
                for i in recoverNodes:
                    self.state[i - 1][1] = 0  # �ָ��ڵ��״̬��ΪS

                # ��¼��ǰʱ��I״̬�ڵ�ı���
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum

            print()

    def spr_cp_raw(self, beita, gama, n, select):
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        edgesNum = len(self.incidenceMatrix[0]) #������ĳ�����
        for repeat in range(self.repeatNum):
            print("\r�ظ�ʵ����ȣ�" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

            # ���� t ��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []  # �ָ��ڵ��б�
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)

                # for node in self.IStateNodes:
                #     chosen_edge = random.choice(self.node_to_edge["v" + str(node)]) #ȡ��I�ڵ���������г���
                #     # print(chosen_edge)
                #     for chosen_node in self.scenes[chosen_edge]:
                #         if self.state[int(chosen_node[1:]) - 1][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                #             if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                #                 self.IStateNodes.append(int(chosen_node[1:]))  # I״̬�ڵ㵱����Ӵ˽ڵ���

                for node, edge_list in enumerate(self.incidenceMatrix, start=1):
                    if self.state[node - 1][1] == 1:  # �ýڵ��Ƿ�ΪI״̬�ڵ㣬�������ѡ��һ�����߽��д�����
                        edges = []
                        for i in range(edgesNum):
                            if edge_list[i] == 1:
                                edges.append('E' + str(i + 1))
                        chosen_edge = random.choice(edges)
                        for chosen_node in self.scenes[chosen_edge]:
                            if self.state[int(chosen_node[1:]) - 1][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                                if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                    self.IStateNodes.append(int(chosen_node[1:]))  # I״̬�ڵ㵱����Ӵ˽ڵ���

                #self.incidenceMatrix ��Ϊ�ڵ㣬��Ϊ����
                # �޸����ݣ�����״̬
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]  # ɾ���ָ��Ľڵ�
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�

                # ����״̬
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1  # ��I״̬�ڵ��״̬��ΪI
                for i in recoverNodes:
                    self.state[i - 1][1] = 0  # �ָ��ڵ��״̬��ΪS

                # ��¼��ǰʱ��I״̬�ڵ�ı���
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum

            print()

    # ��Ϣ����
    def spr1(self, beita, gama, n, select):
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        for repeat in range(self.repeatNum):
            print("\r�ظ�ʵ����ȣ�" + str(repeat + 1)+"/"+str(self.repeatNum), end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

            # ����t��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)
                # ɸѡ��I״̬�ڵ�ĳ���
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 1:  # �˳��ߵ�����I״̬�ڵ�
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                                    if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                        self.IStateNodes.append(int(i[1:]))  # I״̬�ڵ㵱����Ӵ˽ڵ���
                # �޸�����
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�
                # print("�ָ��ڵ㣺",recoverNodes)
                # print("��Ⱦ�ڵ㣺",self.IStateNodes)
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1
                for i in recoverNodes:
                    self.state[i - 1][1] = 0
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum
                # else:
                #     self.repeatNum -= 1
                #     break
        print()

    def spr1_cp(self, beita, gama, n, select):
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        edgesNum = len(self.incidenceMatrix[0]) #������ĳ�����
        for repeat in range(self.repeatNum):
            print("\r�ظ�ʵ����ȣ�" + str(repeat + 1)+"/"+str(self.repeatNum), end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

            # ����t��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)
                # ɸѡ��I״̬�ڵ�ĳ���
                for node, edge_list in enumerate(self.incidenceMatrix, start=1):
                    if self.state[node - 1][1] == 1:  # �ýڵ��Ƿ�ΪI״̬�ڵ㣬�������ѡ��һ�����߽��д�����
                        edges = []
                        for i in range(edgesNum):
                            if edge_list[i] == 1:
                                edges.append('E' + str(i + 1))
                        chosen_edge = random.choice(edges)
                        for chosen_node in self.scenes[chosen_edge]:
                            if self.state[int(chosen_node[1:]) - 1][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                                if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                    self.IStateNodes.append(int(chosen_node[1:]))  # I״̬�ڵ㵱����Ӵ˽ڵ���
                # �޸�����
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�
                # print("�ָ��ڵ㣺",recoverNodes)
                # print("��Ⱦ�ڵ㣺",self.IStateNodes)
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1
                for i in recoverNodes:
                    self.state[i - 1][1] = 0
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum
                # else:
                #     self.repeatNum -= 1
                #     break
        print()

    # ��Ϣ����
    def sprSteadyState(self, beita, gama, n, select):
        self.imageData = np.zeros((self.timeStep, 20), float)  # ���ÿ��ʱ�䲽�Ħѵ��������
        # �ظ�n��ʵ��
        selectnode = 0  # ��¼��ʼ�����ڵ�
        for repeat in range(self.repeatNum):
            # print("\r�ظ�ʵ����ȣ�" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # ��ʼ��
            self.IStateNodes = []  # ���I״̬�ڵ���
            self.state = np.zeros((self.nodesNum, 2), int)  # �ڵ㣬״̬(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0��ʾS״̬

            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # ���ѡ��ڵ���Ϊ�����ڵ�
                elif select == "avgHi":
                    selectnode = self.avgHi()  # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
                elif select == "maxHi":
                    selectnode = self.maxHi()  # ѡ�񳬶����ڵ���Ϊ�����ڵ�
                elif select == "minHi":
                    selectnode = self.minHi()  # ѡ�񳬶���С�ڵ���Ϊ�����ڵ�
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1��ʾI״̬
            edgesNum = len(self.incidenceMatrix[0])  # ������ĳ�����
            # ����t��ʱ�䲽����
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I�ָ�ΪS
                            recoverNodes.append(i)
                # ɸѡ��I״̬�ڵ�ĳ���
                for e in range(edgesNum):
                    for node in self.scenes['E' + str(e + 1)]:
                        node_id = int(node[1:]) - 1
                        if self.state[node_id][1] == 1:  # �˳��ߵ�����I״̬�ڵ�
                            for i in self.scenes['E' + str(e + 1)]:
                                i_id = int(i[1:]) - 1
                                if self.state[i_id][1] == 0:  # ����S̬�Ľڵ���һ�����ʸ�Ⱦ
                                    if random.randint(1, 100) <= 100 * beita:  # ��Ⱦ
                                        self.IStateNodes.append(i_id + 1)  # I״̬�ڵ㵱����Ӵ˽ڵ���
                # �޸�����
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # ȥ���ظ��ڵ�
                # print("�ָ��ڵ㣺",recoverNodes)
                # print("��Ⱦ�ڵ㣺",self.IStateNodes)
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1
                for i in recoverNodes:
                    self.state[i - 1][1] = 0
                self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum
        # print(self.imageData[self.timeStep-1][n])
        avg = 0
        for i in range(1,11):
            avg += self.imageData[self.timeStep-i][n]/10
        self.stateData.append(avg)
        # print(self.stateData)
        # print()

    # ���ѡ��ڵ���Ϊ�����ڵ�
    def randomHi(self):
        # ���ѡ���ʼһ��I״̬�ڵ�
        selectnode = random.randint(1, self.nodesNum)
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

    # ѡ��ƽ�����Ƚڵ���Ϊ�����ڵ�
    def avgHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        avg = (max(degreeList) + min(degreeList)) // 2
        while avg not in degreeList:
            avg -= 1
        selectnode = degreeList.index(avg) + 1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

    # ѡ�񳬶����Ľڵ���Ϊ�����ڵ�
    def maxHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(max(degreeList))+1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

    # ѡ�񳬶���С�Ľڵ���Ϊ�����ڵ�
    def minHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(min(degreeList)) + 1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1��ʾI״̬

    # ��ͼ
    def imgDraw(self):
        t = [i/100 for i in range(0, self.betaNum)]
        yvalue0 = []
        # yvalue1 = [0]
        # yvalue2 = [0]
        # yvalue3 = [0]
        # yvalue4 = [0]
        # yvalue5 = [0]
        for i in range(self.betaNum):
            yvalue0.append(self.stateData[i])
            # yvalue1.append(self.imageData[i][1])
            # yvalue2.append(self.imageData[i][2])
            # yvalue3.append(self.imageData[i][3])
            # yvalue4.append(self.imageData[i][4])
            # yvalue5.append(self.imageData[i][5])
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,)
        #ax.legend(labels=[r"$��=0.05$", r"$��=0.1$", r"$��=0.2$"], ncol=1, fontsize=20)
        #plt.figure("��ͬ�µ�֪��ڵ��ܶ�ͼ", figsize=(10, 8))
        plt.xlabel("��", fontsize=15)
        plt.ylabel("��", fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        #plt.legend(["p=0.6",])
        # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw2(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["N=1000","N=3000","N=5000", ])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # ax.plot(t, yvalue0, color='k', linestyle='-', linewidth=1, marker='s', markersize=7,
        #         )
        # ax.plot(t, yvalue1, color='b', linestyle='-', linewidth=1, marker='d', markersize=7,
        #         )
        # ax.plot(t, yvalue2, color='r', linestyle='-', linewidth=1, marker='^', markersize=7,
        #         )
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )
        # ax.plot(t, yvalue0, color='#C6B3D3', linestyle='-', linewidth=1, marker='s', markersize=8,
        #         )
        # ax.plot(t, yvalue1, color='#ED9F9B', linestyle='-', linewidth=1, marker='d', markersize=8,
        #         )
        # ax.plot(t, yvalue2, color='#80BA8A', linestyle='-', linewidth=1, marker='p', markersize=8,
        #         )

        ax.legend(labels=[r"$N=1000$",r"$N=3000$",r"$N=5000$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw3(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["��=0.05", "��=0.1", "��=0.2", ])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$��=0.05$", r"$��=0.1$", r"$��=0.2$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw4(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["��=0.05", "��=0.10", "��=0.20", ])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$��=0.05$", r"$��=0.10$", r"$��=0.20$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw5(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]
        # yvalue3 = [0]
        # yvalue4 = [0]
        # yvalue5 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])
            # yvalue3.append(self.imageData[i][3])
            # yvalue4.append(self.imageData[i][4])
            # yvalue5.append(self.imageData[i][5])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          t, yvalue3, '-o',
        #          t, yvalue4, '-p',
        #          t, yvalue5, '-*',
        #          )
        #
        # plt.legend(["p=0", "p=0.2", "p=0.4", "p=0.6","p=0.8", "p=1" ])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )
        # ax.plot(t, yvalue3, color='#d62728', linestyle='-', linewidth=1, marker='^', markersize=8,
        #         )
        # ax.plot(t, yvalue4, color='#9467bd', linestyle='-', linewidth=1, marker='o', markersize=8,
        #         )
        # ax.plot(t, yvalue5, color='#8c564b', linestyle='-', linewidth=1, marker='*', markersize=8,
        #         )
        # ax.plot(t, yvalue0, color='#C6B3D3', linestyle='-', linewidth=1, marker='s', markersize=8,
        #         )
        # ax.plot(t, yvalue1, color='#ED9F9B', linestyle='-', linewidth=1, marker='d', markersize=8,
        #         )
        # ax.plot(t, yvalue2, color='#80BA8A', linestyle='-', linewidth=1, marker='p', markersize=8,
        #         )
        # ax.plot(t, yvalue3, color='#9CD1C8', linestyle='-', linewidth=1, marker='^', markersize=8,
        #         )
        # ax.plot(t, yvalue4, color='#6BB7CA', linestyle='-', linewidth=1, marker='o', markersize=8,
        #         )
        # ax.plot(t, yvalue5, color='#F7D58B', linestyle='-', linewidth=1, marker='*', markersize=8,
        #         )
        # ax.legend(labels=["p=0", "p=0.2", "p=0.4", "p=0.6","p=0.8", "p=1"], ncol=3)
        ax.legend(loc=(0.75,0.48),labels=[r"$n=2$", r"$n=3$", r"$n=4$"], ncol=1, fontsize=18)

        # ax:������ϵ��width,height:������ϵ�Ŀ�Ⱥ͸߶�(�ٷֱ���ʽ���߸���������)��loc:������ϵ��λ�ã�
        # bbox_to_anchor:�߽����Ԫ����(x0,y0,width,height);bbox_transform:�Ӹ�����ϵ��������ϵ�ļ���ӳ��;
        # axins:������ϵ
        # axins = inset_axes(ax,width="40%", height="30%", loc='lower left',
        #                    bbox_to_anchor=(0.1, 0.1, 1, 1),bbox_transform=ax.transAxes)
        axins = ax.inset_axes((0.6, 0.05, 0.4, 0.4))  # ��ͼλ��
        # axins = ax.inset_axes((0.3, 0.05, 0.4, 0.4))  # ��ͼλ��
        axins.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=5,
                   )
        axins.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=5,
                   )
        axins.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=5,
                   )
        # axins.plot(t, yvalue3, color='#d62728', linestyle='-', linewidth=1, marker='^', markersize=5,
        #            )
        # axins.plot(t, yvalue4, color='#9467bd', linestyle='-', linewidth=1, marker='o', markersize=5,
        #            )
        # axins.plot(t, yvalue5, color='#8c564b', linestyle='-', linewidth=1, marker='*', markersize=5,
        #            )
        # ����������ϵ����ʾ��Χ
        axins.set_xlim(1, 5)
        axins.set_ylim(0.3, 0.5)
        #plt.xticks(np.arange(0, 21, step=5))
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        # ����������ϵ��������ϵ��������
        # loc1 loc2: ����ϵ���ĸ���
        # 1 (����) 2 (����) 3(����) 4(����)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw6(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend([ "maxDi","avgDi", "minDi"])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # ax.plot(t, yvalue0, color='k', linestyle='-', linewidth=1, marker='s', markersize=7,
        #         )
        # ax.plot(t, yvalue1, color='b', linestyle='-', linewidth=1, marker='d', markersize=7,
        #         )
        # ax.plot(t, yvalue2, color='r', linestyle='-', linewidth=1, marker='^', markersize=7,
        #         )
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )
        # ax.plot(t, yvalue0, color='#C6B3D3', linestyle='-', linewidth=1, marker='s', markersize=8,
        #         )
        # ax.plot(t, yvalue1, color='#ED9F9B', linestyle='-', linewidth=1, marker='d', markersize=8,
        #         )
        # ax.plot(t, yvalue2, color='#80BA8A', linestyle='-', linewidth=1, marker='p', markersize=8,
        #         )

        ax.legend(labels=["minHD","avgHD","maxHD"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw7(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-d',
        #          t, yvalue1, '-s',
        #          )
        #
        # plt.legend(["Clustering Hypernetwork", "BA Hypernetwork"])
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )

        # ax.legend(labels=["Aggregation Phenomenon Hypernetwork", "BA Hypernetwork"], ncol=3)
        #ax.legend(labels=["Local Stochastically Variable Scale-free Hypernetwork", "BA Hypernetwork"], ncol=1, fontsize=18)
        # ax.legend(labels=["Local Stochastically Variable Scale-free Hypernetwork", "BA Hypernetwork"], ncol=1, fontsize=18, loc='lower right', bbox_to_anchor=(1, 0))
        ax.legend(labels=["Local Stochastically Variable Scale-free Hypernetwork", "BA Hypernetwork"], ncol=1, fontsize=16, loc='lower right')
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw8(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]
        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])
            if i==self.timeStep-1:
                print("beta",self.beta,":",self.imageData[i][0],self.imageData[i][1],self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        # plt.legend(["m1=1", "m1=3", "m1=5"])
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        # ax.legend(labels=["m1=1", "m1=3", "m1=5"], ncol=3)

        # ax:������ϵ��width,height:������ϵ�Ŀ�Ⱥ͸߶�(�ٷֱ���ʽ���߸���������)��loc:������ϵ��λ�ã�
        # bbox_to_anchor:�߽����Ԫ����(x0,y0,width,height);bbox_transform:�Ӹ�����ϵ��������ϵ�ļ���ӳ��;
        # axins:������ϵ
        # axins = inset_axes(ax,width="40%", height="30%", loc='lower left',
        #                    bbox_to_anchor=(0.1, 0.1, 1, 1),bbox_transform=ax.transAxes)
        # axins = ax.inset_axes((0.4, 0.1, 0.4, 0.4))  # ��ͼλ��
        # axins.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=5)
        # axins.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=5)
        # axins.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=5)

        # ����������ϵ����ʾ��Χ
        # axins.set_xlim(0, 5)
        # axins.set_ylim(0.35, 0.5)
        # plt.xticks(np.arange(0, 20, step=5))

        ax.legend(loc=(0.75, 0.55), labels=[r"$m_1=1$", r"$m_1=3$", r"$m_1=5$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        # ����������ϵ��������ϵ��������
        # loc1 loc2: ����ϵ���ĸ���
        # 1 (����) 2 (����) 3(����) 4(����)
        # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw9(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["m2=1", "m2=3", "m2=5"])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$m_2=1$", r"$m_2=3$", r"$m_2=5$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw10(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["m=1", "m=3", "m=5"])
        # # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$m=1$", r"$m=3$", r"$m=5$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw11(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("��")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # # plt.legend(["m1=1,m2=1", "m1=3,m2=3", "m1=5,m2=5"])
        # plt.legend(["m1=1,m2=5", "m1=3,m2=3", "m1=5,m2=1"])
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$m_1=1,m_2=5$", r"$m_1=3,m_2=3$", r"$m_1=5,m_2=1$"], ncol=1, fontsize=20)
        # ax.legend(labels=[r"$m_1=1,m_2=1$", r"$m_1=3,m_2=3$", r"$m_1=5,m_2=5$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("��", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # �޸Ŀ̶����ߴ�ϸwidth����
        ax.spines['bottom'].set_linewidth(1.5)  ###���õײ�������Ĵ�ϸ
        ax.spines['left'].set_linewidth(1.5)  ####�������������Ĵ�ϸ
        ax.spines['right'].set_linewidth(1.5)  ###�����ұ�������Ĵ�ϸ
        ax.spines['top'].set_linewidth(1.5)

        # plt.xticks(np.arange(0, 20, step=5))

        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw12(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]
        yvalue3 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])
            yvalue3.append(self.imageData[i][3])

        plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        plt.xlabel("t")
        plt.ylabel("��")
        plt.plot(t, yvalue0, '-s',
                 t, yvalue1, '-d',
                 t, yvalue2, '-^',
                 t, yvalue3, '-o',
                 )

        plt.legend(["m2=1,p=0.2", "m2=3,p=0.2", "m2=1,p=0.8", "m2=3,p=0.8"])
        # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ��ͼ
    def imgDraw13(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]
        yvalue3 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])
            yvalue3.append(self.imageData[i][3])

        plt.figure("��ͬģ��֪ʶ��ɢͼ", figsize=(10, 8))
        plt.xlabel("t")
        plt.ylabel("��")
        plt.plot(t, yvalue0, '-s',
                 t, yvalue1, '-d',
                 t, yvalue2, '-^',
                 t, yvalue3, '-o',
                 )

        plt.legend(["m1=1,p=0.2", "m1=3,p=0.2", "m1=1,p=0.8", "m1=3,p=0.8"])
        # plt.rcParams['savefig.dpi'] = 100  # ͼƬ����
        plt.savefig("img.svg", format='svg', dpi=600)  # svg��ʽ
        plt.show()

    # ���ݴ洢
    def dataExport(self):
        print("�洢...")
        with open(self.dataSave + 'imgData.txt', 'w') as file0:
            print("yData:", file=file0)
            print(self.imageData, file=file0)
        print()

if __name__ == '__main__':
    infospr = InfoSpreading()
    warnings.filterwarnings("error")
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    #infospr.funRun1()  # ��̬
    # infospr.funRun2()  # ��ͬ�ڵ���N
    # infospr.funRun3()  # ��ͬ�����ʦ�
    # infospr.funRun4()  # ��ͬ�ָ��ʦ�
    infospr.funRun5()  # ��ͬ�ھӽ���n

    # infospr.importModel_raw(1000, 3, 3, 3)
    # infospr.spr(0.1, 0.1, 0, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 2)
    # infospr.spr(0.1, 0.1, 1, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 3)
    # infospr.spr(0.1, 0.1, 2, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 4)
    # infospr.spr(0.1, 0.1, 3, "avgHi")
    # self.dataExport()  # ���ݴ洢
    # infospr.imgDraw5()  # ��ͼ

    # infospr.funRun6()  # ��ͬ��ʼ�����ڵ�
    # infospr.funRun7()  # ��BA������Ա�
    # infospr.funRun8()  # ��ͬm1
    # infospr.funRun9()  # ��ͬm2
    # infospr.funRun10()  # ��ͬm
    # infospr.funRun11()  # m1+m2��Ӱ��
    # infospr.funRun12()  # ��ͬp,m2�仯��Ӱ��
    # infospr.funRun13()  # ��ͬp,m1�仯��Ӱ��
