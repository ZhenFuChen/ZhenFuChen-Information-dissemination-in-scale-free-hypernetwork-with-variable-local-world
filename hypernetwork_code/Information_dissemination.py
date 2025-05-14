#coding=gbk
import random
import numpy as np
import hypernetx as hnx
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
# 将警告级别设置为 ERROR，意味着只会显示 ERROR 级别的警告
class InfoSpreading(object):
    def __init__(self):
        self.nodesNum = 0
        self.beta = 0
        self.gamma = 0
        self.timeStep = 50 # 时间步
        self.repeatNum = 100  # 重复实验次数(默认100)
        self.betaNum = 21
        self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
        self.imageData = np.zeros((self.timeStep, 20), float)  # 存放每个时间步的ρ等相关数据
        self.stateData = []
        self.scenes = {}
        self.node_to_edge = {}
        self.imgSave = "../Save/ImgSave/"
        self.importPath = "ModelSave/"
        self.dataSave ="../Save/DataSave/ResultData/"

    # 稳态下ρ的仿真
    def funRun1(self):
        # 导入模型
        self.importModel(1000,3,3,3,2)
        for beta in range(self.betaNum):
            print("\rβ="+str(beta/100),end="")
            self.sprSteadyState(beta/100, 0.3, 0, "avgHi")
        print(self.stateData)
        #self.dataExport()  # 数据存储
        self.imgDraw()  # 绘图

    # 不同节点数
    def funRun2(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000,3,3,3,2)
        self.spr(0.1,0.1,0, "avgHi")
        self.importModel(3000,3,3,3,2)
        self.spr(0.1,0.1,1, "avgHi")
        self.importModel(5000,3,3,3,2)
        self.spr(0.1,0.1,2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw2()  # 绘图

    # 不同传播率β
    def funRun3(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.05,0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp_raw(0.2, 0.1, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw3()  # 绘图

    # 不同恢复率γ
    def funRun4(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.05, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.10, 1, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.20, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw4()  # 绘图

    # 不同邻居阶数n
    def funRun5(self):
        # 导入模型
        # 超边变量增长
        # self.importModel_raw(1000,3,3,3)
        # self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,3)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,3,3,3,4)
        self.spr(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw5()  # 绘图

    # 不同初始传播节点
    def funRun6(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 0, "minHi")
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3,3,3,2)
        self.spr(0.1, 0.1, 2, "maxHi")

        # self.dataExport()  # 数据存储
        self.imgDraw6()  # 绘图

    # 与BA超网络对比
    def funRun7(self):
        # 导入模型
        # 超边变量增长
        self.importModel_raw(1000,3,3,3)
        self.spr(0.1, 0.1, 0, "avgHi")
        self.importModel_raw(1000,3,3,3)
        self.spr(0.1, 0.1, 1, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw7()  # 绘图

    # 不同m1
    def funRun8(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000,1,3,3,2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000,3,3,3,2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000,5,3,3,2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # 下方代码理论与实际曲线绘图时使用，修改beta
        # self.importModel(1000, 1, 3, 3)
        # self.spr(0.2, 0.1, 0, "avgHi")
        # self.importModel(1000, 3, 3, 3)
        # self.spr(0.2, 0.1, 1, "avgHi")
        # self.importModel(1000, 5, 3, 3)
        # self.spr(0.2, 0.1, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw8()  # 绘图

    # 不同m2
    def funRun9(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000, 3, 1, 3, 2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000, 3, 3, 3, 2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3, 5, 3, 2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw9()  # 绘图

    # 不同m
    def funRun10(self):
        # 导入模型
        # 超边变量增长
        self.importModel(1000, 3, 3, 1, 2)
        self.spr_cp(0.1, 0.1, 0, "avgHi")
        self.importModel(1000, 3, 3, 3, 2)
        self.spr_cp(0.1, 0.1, 1, "avgHi")
        self.importModel(1000, 3, 3, 5, 2)
        self.spr_cp(0.1, 0.1, 2, "avgHi")

        # self.dataExport()  # 数据存储
        self.imgDraw10()  # 绘图

    # m1+m2的影响
    def funRun11(self):
        # 导入模型
        # 超边变量增长
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

        # self.dataExport()  # 数据存储
        self.imgDraw11()  # 绘图

    # 不同p,m2变化的影响
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

        self.imgDraw12()  # 绘图

    # 不同p,m1变化的影响
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

        self.imgDraw13()  # 绘图

    # 导入数据
    def importModel(self, N,m1,m2,m,n):  # 节点数(1000,3000,5000),超边数,θ值(实际值的十倍),模型(edge,node)
        self.nodesNum = N
        self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
        self.scenes = {}
        print("导入n" + str(N) +"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+"n"+str(n) + "模型")
        # 超网络模型导入
        self.incidenceMatrix = np.genfromtxt(self.importPath+"n"+str(self.nodesNum)+"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+"_"+str(n)+"version.txt",delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # 超边中的节点用元组存储
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
            self.scenes["E" + str(i + 1)] = eTuple
        edgesNum = len(self.incidenceMatrix[0]) #超网络的超边数
        for node, edge_list in enumerate(self.incidenceMatrix, start=1):
            edges = ()
            for i in range(edgesNum):
                if edge_list[i] != 0:
                    edges += ('E' + str(i + 1),)
            self.node_to_edge["v" + str(node)] = edges

    def importModel_raw(self, N,m1,m2,m):  # 节点数(1000,3000,5000),超边数,θ值(实际值的十倍),模型(edge,node)
        self.nodesNum = N
        self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
        self.scenes = {}
        print("导入n" + str(N) +"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+ "模型")
        # 超网络模型导入
        self.incidenceMatrix = np.genfromtxt(self.importPath+"n"+str(self.nodesNum)+"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+".txt",delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # 超边中的节点用元组存储
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
                    # if len(eTuple) == 4:
                    #     break
            self.scenes["E" + str(i + 1)] = eTuple

    # 导入数据
    def importModel1(self, N, E, theta, m1,m2,m):  # 节点数(1000,3000,5000),超边数,θ值(实际值的十倍),模型(edge,node)
        self.nodesNum = N
        self.theta = theta
        self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
        self.scenes = {}
        print("正在导入n{}e{}theta{}m1{}m2{}m{}的模型".format(N,E,theta,m1,m2,m))
        # 超网络模型导入
        self.incidenceMatrix = np.genfromtxt(
            self.importPath + "n" + str(N) + "e" + str(E) + "theta" + str(
                theta) + "m1"+str(m1)+"m2"+str(m2)+"m"+str(m) + "EdgeInc.txt", delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # 超边中的节点用元组存储
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
                    # if len(eTuple) == 4:
                    #     break
            self.scenes["E" + str(i + 1)] = eTuple

    # 信息传播
    def spr(self, beita, gama, n, select):
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        for repeat in range(self.repeatNum):
            print("\r重复实验进度："+str(repeat+1)+"/"+str(self.repeatNum),end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态

            edgesNum = len(self.incidenceMatrix[0])  # 超网络的超边数
            # 进行t个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)
                # 筛选有I状态节点的超边
                for e in range(edgesNum):
                    for node in self.scenes['E' + str(e + 1)]:
                        node_id = int(node[1:]) - 1
                        if self.state[node_id][1] == 1:  # 此超边当中有I状态节点
                            for i in self.scenes['E' + str(e + 1)]:
                                i_id = int(i[1:]) - 1
                                if self.state[i_id][1] == 0:  # 处于S态的节点以一定概率感染
                                    if random.randint(1, 100) <= 100 * beita:  # 感染
                                        self.IStateNodes.append(i_id + 1)  # I状态节点当中添加此节点编号
                # 修改数据
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点
                # print("恢复节点：",recoverNodes)
                # print("感染节点：",self.IStateNodes)
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1
                for i in recoverNodes:
                    self.state[i - 1][1] = 0
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum
        print()

    def spr_cp(self, beita, gama, n, select):
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        for repeat in range(self.repeatNum):
            print("\r重复实验进度：" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态

            # 进行 t 个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []  # 恢复节点列表
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)

                for node in self.IStateNodes:
                    chosen_edge = random.choice(self.node_to_edge["v" + str(node)]) #取出I节点关联的所有超边
                    # print(chosen_edge)
                    for chosen_node in self.scenes[chosen_edge]:
                        if self.state[int(chosen_node[1:]) - 1][1] == 0:  # 处于S态的节点以一定概率感染
                            if random.randint(1, 100) <= 100 * beita:  # 感染
                                self.IStateNodes.append(int(chosen_node[1:]))  # I状态节点当中添加此节点编号

                #self.incidenceMatrix 行为节点，列为超边
                # 修改数据，更新状态
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]  # 删除恢复的节点
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点

                # 更新状态
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1  # 将I状态节点的状态设为I
                for i in recoverNodes:
                    self.state[i - 1][1] = 0  # 恢复节点的状态设为S

                # 记录当前时刻I状态节点的比例
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum

            print()

    def spr_cp_raw(self, beita, gama, n, select):
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        edgesNum = len(self.incidenceMatrix[0]) #超网络的超边数
        for repeat in range(self.repeatNum):
            print("\r重复实验进度：" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态

            # 进行 t 个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []  # 恢复节点列表
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)

                # for node in self.IStateNodes:
                #     chosen_edge = random.choice(self.node_to_edge["v" + str(node)]) #取出I节点关联的所有超边
                #     # print(chosen_edge)
                #     for chosen_node in self.scenes[chosen_edge]:
                #         if self.state[int(chosen_node[1:]) - 1][1] == 0:  # 处于S态的节点以一定概率感染
                #             if random.randint(1, 100) <= 100 * beita:  # 感染
                #                 self.IStateNodes.append(int(chosen_node[1:]))  # I状态节点当中添加此节点编号

                for node, edge_list in enumerate(self.incidenceMatrix, start=1):
                    if self.state[node - 1][1] == 1:  # 该节点是否为I状态节点，是则随机选择一条超边进行传播。
                        edges = []
                        for i in range(edgesNum):
                            if edge_list[i] == 1:
                                edges.append('E' + str(i + 1))
                        chosen_edge = random.choice(edges)
                        for chosen_node in self.scenes[chosen_edge]:
                            if self.state[int(chosen_node[1:]) - 1][1] == 0:  # 处于S态的节点以一定概率感染
                                if random.randint(1, 100) <= 100 * beita:  # 感染
                                    self.IStateNodes.append(int(chosen_node[1:]))  # I状态节点当中添加此节点编号

                #self.incidenceMatrix 行为节点，列为超边
                # 修改数据，更新状态
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]  # 删除恢复的节点
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点

                # 更新状态
                for i in self.IStateNodes:
                    self.state[i - 1][1] = 1  # 将I状态节点的状态设为I
                for i in recoverNodes:
                    self.state[i - 1][1] = 0  # 恢复节点的状态设为S

                # 记录当前时刻I状态节点的比例
                if len(self.IStateNodes) != 0:
                    self.imageData[t][n] += len(self.IStateNodes) / self.nodesNum / self.repeatNum

            print()

    # 信息传播
    def spr1(self, beita, gama, n, select):
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        for repeat in range(self.repeatNum):
            print("\r重复实验进度：" + str(repeat + 1)+"/"+str(self.repeatNum), end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态

            # 进行t个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)
                # 筛选有I状态节点的超边
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 1:  # 此超边当中有I状态节点
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:  # 处于S态的节点以一定概率感染
                                    if random.randint(1, 100) <= 100 * beita:  # 感染
                                        self.IStateNodes.append(int(i[1:]))  # I状态节点当中添加此节点编号
                # 修改数据
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点
                # print("恢复节点：",recoverNodes)
                # print("感染节点：",self.IStateNodes)
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
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        edgesNum = len(self.incidenceMatrix[0]) #超网络的超边数
        for repeat in range(self.repeatNum):
            print("\r重复实验进度：" + str(repeat + 1)+"/"+str(self.repeatNum), end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态
            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态

            # 进行t个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)
                # 筛选有I状态节点的超边
                for node, edge_list in enumerate(self.incidenceMatrix, start=1):
                    if self.state[node - 1][1] == 1:  # 该节点是否为I状态节点，是则随机选择一条超边进行传播。
                        edges = []
                        for i in range(edgesNum):
                            if edge_list[i] == 1:
                                edges.append('E' + str(i + 1))
                        chosen_edge = random.choice(edges)
                        for chosen_node in self.scenes[chosen_edge]:
                            if self.state[int(chosen_node[1:]) - 1][1] == 0:  # 处于S态的节点以一定概率感染
                                if random.randint(1, 100) <= 100 * beita:  # 感染
                                    self.IStateNodes.append(int(chosen_node[1:]))  # I状态节点当中添加此节点编号
                # 修改数据
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点
                # print("恢复节点：",recoverNodes)
                # print("感染节点：",self.IStateNodes)
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

    # 信息传播
    def sprSteadyState(self, beita, gama, n, select):
        self.imageData = np.zeros((self.timeStep, 20), float)  # 存放每个时间步的ρ等相关数据
        # 重复n次实验
        selectnode = 0  # 记录初始传播节点
        for repeat in range(self.repeatNum):
            # print("\r重复实验进度：" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            # 初始化
            self.IStateNodes = []  # 存放I状态节点编号
            self.state = np.zeros((self.nodesNum, 2), int)  # 节点，状态(S(0),I(1))
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0  # 0表示S状态

            if repeat == 0:
                if select == "random":
                    selectnode = self.randomHi()  # 随机选择节点作为传播节点
                elif select == "avgHi":
                    selectnode = self.avgHi()  # 选择平均超度节点作为传播节点
                elif select == "maxHi":
                    selectnode = self.maxHi()  # 选择超度最大节点作为传播节点
                elif select == "minHi":
                    selectnode = self.minHi()  # 选择超度最小节点作为传播节点
            self.IStateNodes.append(selectnode)
            self.state[selectnode - 1][1] = 1  # 1表示I状态
            edgesNum = len(self.incidenceMatrix[0])  # 超网络的超边数
            # 进行t个时间步传播
            for t in range(self.timeStep):
                recoverNodes = []
                if t >= 1:
                    for i in self.IStateNodes:
                        if random.randint(1, 100) <= 100 * gama:  # I恢复为S
                            recoverNodes.append(i)
                # 筛选有I状态节点的超边
                for e in range(edgesNum):
                    for node in self.scenes['E' + str(e + 1)]:
                        node_id = int(node[1:]) - 1
                        if self.state[node_id][1] == 1:  # 此超边当中有I状态节点
                            for i in self.scenes['E' + str(e + 1)]:
                                i_id = int(i[1:]) - 1
                                if self.state[i_id][1] == 0:  # 处于S态的节点以一定概率感染
                                    if random.randint(1, 100) <= 100 * beita:  # 感染
                                        self.IStateNodes.append(i_id + 1)  # I状态节点当中添加此节点编号
                # 修改数据
                self.IStateNodes = [i for i in self.IStateNodes if i not in recoverNodes]
                self.IStateNodes = list(set(self.IStateNodes))  # 去除重复节点
                # print("恢复节点：",recoverNodes)
                # print("感染节点：",self.IStateNodes)
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

    # 随机选择节点作为传播节点
    def randomHi(self):
        # 随机选择初始一个I状态节点
        selectnode = random.randint(1, self.nodesNum)
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1表示I状态

    # 选择平均超度节点作为传播节点
    def avgHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        avg = (max(degreeList) + min(degreeList)) // 2
        while avg not in degreeList:
            avg -= 1
        selectnode = degreeList.index(avg) + 1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1表示I状态

    # 选择超度最大的节点作为传播节点
    def maxHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(max(degreeList))+1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1表示I状态

    # 选择超度最小的节点作为传播节点
    def minHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(min(degreeList)) + 1
        return selectnode
        # self.IStateNodes.append(selectnode)
        # self.state[selectnode - 1][1] = 1  # 1表示I状态

    # 绘图
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
        #ax.legend(labels=[r"$β=0.05$", r"$β=0.1$", r"$β=0.2$"], ncol=1, fontsize=20)
        #plt.figure("不同β的知情节点密度图", figsize=(10, 8))
        plt.xlabel("β", fontsize=15)
        plt.ylabel("ρ", fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        #plt.legend(["p=0.6",])
        # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw2(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["N=1000","N=3000","N=5000", ])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw3(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["β=0.05", "β=0.1", "β=0.2", ])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$β=0.05$", r"$β=0.1$", r"$β=0.2$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw4(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["γ=0.05", "γ=0.10", "γ=0.20", ])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        ax.legend(labels=[r"$γ=0.05$", r"$γ=0.10$", r"$γ=0.20$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
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

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          t, yvalue3, '-o',
        #          t, yvalue4, '-p',
        #          t, yvalue5, '-*',
        #          )
        #
        # plt.legend(["p=0", "p=0.2", "p=0.4", "p=0.6","p=0.8", "p=1" ])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
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

        # ax:父坐标系；width,height:子坐标系的宽度和高度(百分比形式或者浮点数个数)；loc:子坐标系的位置；
        # bbox_to_anchor:边界框，四元数组(x0,y0,width,height);bbox_transform:从父坐标系到子坐标系的几何映射;
        # axins:子坐标系
        # axins = inset_axes(ax,width="40%", height="30%", loc='lower left',
        #                    bbox_to_anchor=(0.1, 0.1, 1, 1),bbox_transform=ax.transAxes)
        axins = ax.inset_axes((0.6, 0.05, 0.4, 0.4))  # 子图位置
        # axins = ax.inset_axes((0.3, 0.05, 0.4, 0.4))  # 子图位置
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
        # 调整子坐标系的显示范围
        axins.set_xlim(1, 5)
        axins.set_ylim(0.3, 0.5)
        #plt.xticks(np.arange(0, 21, step=5))
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw6(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1,self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend([ "maxDi","avgDi", "minDi"])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw7(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-d',
        #          t, yvalue1, '-s',
        #          )
        #
        # plt.legend(["Clustering Hypernetwork", "BA Hypernetwork"])
        # plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
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

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        # plt.legend(["m1=1", "m1=3", "m1=5"])
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        # plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=8,
                )
        ax.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=8,
                )
        ax.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=8,
                )

        # ax.legend(labels=["m1=1", "m1=3", "m1=5"], ncol=3)

        # ax:父坐标系；width,height:子坐标系的宽度和高度(百分比形式或者浮点数个数)；loc:子坐标系的位置；
        # bbox_to_anchor:边界框，四元数组(x0,y0,width,height);bbox_transform:从父坐标系到子坐标系的几何映射;
        # axins:子坐标系
        # axins = inset_axes(ax,width="40%", height="30%", loc='lower left',
        #                    bbox_to_anchor=(0.1, 0.1, 1, 1),bbox_transform=ax.transAxes)
        # axins = ax.inset_axes((0.4, 0.1, 0.4, 0.4))  # 子图位置
        # axins.plot(t, yvalue0, color='#1f77b4', linestyle='-', linewidth=1, marker='s', markersize=5)
        # axins.plot(t, yvalue1, color='#ff7f0e', linestyle='-', linewidth=1, marker='d', markersize=5)
        # axins.plot(t, yvalue2, color='#0ca022', linestyle='-', linewidth=1, marker='p', markersize=5)

        # 调整子坐标系的显示范围
        # axins.set_xlim(0, 5)
        # axins.set_ylim(0.35, 0.5)
        # plt.xticks(np.arange(0, 20, step=5))

        ax.legend(loc=(0.75, 0.55), labels=[r"$m_1=1$", r"$m_1=3$", r"$m_1=5$"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw9(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["m2=1", "m2=3", "m2=5"])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw10(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # plt.legend(["m=1", "m=3", "m=5"])
        # # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
    def imgDraw11(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [0]
        yvalue1 = [0]
        yvalue2 = [0]

        for i in range(1, self.timeStep):
            yvalue0.append(self.imageData[i][0])
            yvalue1.append(self.imageData[i][1])
            yvalue2.append(self.imageData[i][2])

        # plt.figure("不同模型知识扩散图", figsize=(10, 8))
        # plt.xlabel("t")
        # plt.ylabel("ρ")
        # plt.plot(t, yvalue0, '-s',
        #          t, yvalue1, '-d',
        #          t, yvalue2, '-^',
        #          )
        #
        # # plt.legend(["m1=1,m2=1", "m1=3,m2=3", "m1=5,m2=5"])
        # plt.legend(["m1=1,m2=5", "m1=3,m2=3", "m1=5,m2=1"])
        # plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
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
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)

        # plt.xticks(np.arange(0, 20, step=5))

        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
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

        plt.figure("不同模型知识扩散图", figsize=(10, 8))
        plt.xlabel("t")
        plt.ylabel("ρ")
        plt.plot(t, yvalue0, '-s',
                 t, yvalue1, '-d',
                 t, yvalue2, '-^',
                 t, yvalue3, '-o',
                 )

        plt.legend(["m2=1,p=0.2", "m2=3,p=0.2", "m2=1,p=0.8", "m2=3,p=0.8"])
        # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 绘图
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

        plt.figure("不同模型知识扩散图", figsize=(10, 8))
        plt.xlabel("t")
        plt.ylabel("ρ")
        plt.plot(t, yvalue0, '-s',
                 t, yvalue1, '-d',
                 t, yvalue2, '-^',
                 t, yvalue3, '-o',
                 )

        plt.legend(["m1=1,p=0.2", "m1=3,p=0.2", "m1=1,p=0.8", "m1=3,p=0.8"])
        # plt.rcParams['savefig.dpi'] = 100  # 图片像素
        plt.savefig("img.svg", format='svg', dpi=600)  # svg格式
        plt.show()

    # 数据存储
    def dataExport(self):
        print("存储...")
        with open(self.dataSave + 'imgData.txt', 'w') as file0:
            print("yData:", file=file0)
            print(self.imageData, file=file0)
        print()

if __name__ == '__main__':
    infospr = InfoSpreading()
    warnings.filterwarnings("error")
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    #infospr.funRun1()  # 稳态
    # infospr.funRun2()  # 不同节点数N
    # infospr.funRun3()  # 不同传播率β
    # infospr.funRun4()  # 不同恢复率γ
    infospr.funRun5()  # 不同邻居阶数n

    # infospr.importModel_raw(1000, 3, 3, 3)
    # infospr.spr(0.1, 0.1, 0, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 2)
    # infospr.spr(0.1, 0.1, 1, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 3)
    # infospr.spr(0.1, 0.1, 2, "avgHi")
    # infospr.importModel(1000, 3, 3, 3, 4)
    # infospr.spr(0.1, 0.1, 3, "avgHi")
    # self.dataExport()  # 数据存储
    # infospr.imgDraw5()  # 绘图

    # infospr.funRun6()  # 不同初始传播节点
    # infospr.funRun7()  # 与BA超网络对比
    # infospr.funRun8()  # 不同m1
    # infospr.funRun9()  # 不同m2
    # infospr.funRun10()  # 不同m
    # infospr.funRun11()  # m1+m2的影响
    # infospr.funRun12()  # 不同p,m2变化的影响
    # infospr.funRun13()  # 不同p,m1变化的影响
