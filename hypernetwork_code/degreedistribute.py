#coding=gbk
import json
import math
import random
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import hypernetx as hnx
class HUHPM(object):
    def __init__(self):
        self.scenes = {}
        self.xdata = []
        self.ydata = []
        self.dataSave = "Save/degreeDistributeDataSave/"  # 数据保存路径
        self.imageSave = "./imagesave.png"  # 图片保存路径
        self.importpath = "ModelSave/"

    def initHyperGraph(self):
        self.scenes = {}

    def funRun(self):

        self.importModel(1000, 3, 3, 3, 1)  # N,m1,m2,m,n
        self.degreeDistribute(1000)
        self.theoretical(3, 3, 3)  # m1,m2,m
        self.imaShow()

    def funRun1(self):
        # self.importModel(5000, 3, 3, 3, 1)  # N,m1,m2,m,p(0,1,2...,10)
        # self.degreeDistribute(5000)
        self.importModel(5000, 3, 3, 3, 2)  # N,m1,m2,m,p(0,1,2...,10)
        self.degreeDistribute(5000)
        self.importModel(5000, 3, 3, 3, 3)  # N,m1,m2,m,p(0,1,2...,10)
        self.degreeDistribute(5000)
        self.importModel(5000, 3, 3, 3, 4)  # N,m1,m2,m,p(0,1,2...,10)
        self.degreeDistribute(5000)
        self.theoretical(3, 3, 3)  # m1,m2,m
        self.imaShow1()

    def degreeDistribute(self,N):
        print("正在计算超度分布")
        self.H = hnx.Hypergraph(self.scenes)
        # 计算超度hk
        hk = hnx.degree_dist(self.H)
        # Pk 用来计算超度分布
        Pk = np.zeros(max(hk),float)
        for i in range(N):
            Pk[hk[i]-1] += 1/N
        x = [i for i in range(1,max(hk)+1)]

        self.xdata.append(x)
        self.ydata.append(Pk)

    def theoretical(self,m1,m2,m):
        Pk = np.zeros(50,float)
        for k in range(2,50):
            # Pk[k] = ((m1+1)/m)*((m/k)**(m1+2))
            Pk[k] = (1/m)*(m1/m2+1)*((m/k)**(m1/m2+2))
        x = [i for i in range(2, 50)]
        self.xdata.append(x)
        self.ydata.append(Pk[2:])

    def imaShow(self):
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
        ax.legend(labels=[r"$n=1$", "theoretical"], ncol=1, fontsize=20)

        plt.savefig("img1.svg",format='svg',dpi=600)  # svg格式
        plt.show()

    def imaShow1(self):
        # plt.figure("超度幂律分布P(k)与k对数关系图", figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        # plt.xlabel("k")
        # plt.ylabel("P(k)")
        plt.scatter(self.xdata[0], self.ydata[0], marker='o', facecolors='none', edgecolors='#1f77b4')
        plt.scatter(self.xdata[1], self.ydata[1], marker='s', facecolors='none', edgecolors='#ff7f0e')
        plt.scatter(self.xdata[2], self.ydata[2], marker='^', facecolors='none', edgecolors='#0ca022')
        # plt.scatter(self.xdata[3], self.ydata[3], marker='*', facecolors='none', edgecolors='#e377c2')
        # plt.scatter(self.xdata[4], self.ydata[4], marker='D', facecolors='none', edgecolors='#e377c2')
        # plt.scatter(self.xdata[5], self.ydata[5], marker='^', facecolors='none', edgecolors='#e377c2')
        # plt.scatter(self.xdata[0], self.ydata[0], marker='o', facecolors='none', edgecolors='#ff7f0e')
        # plt.scatter(self.xdata[1], self.ydata[1], marker='s', facecolors='none', edgecolors='#0ca022')
        # plt.scatter(self.xdata[2], self.ydata[2], marker='P', facecolors='none', edgecolors='#d62728')
        # plt.scatter(self.xdata[3], self.ydata[3], marker='*', facecolors='none', edgecolors='#9467bd')
        # plt.scatter(self.xdata[4], self.ydata[4], marker='D', facecolors='none', edgecolors='#8c564b')
        # plt.scatter(self.xdata[5], self.ydata[5], marker='^', facecolors='none', edgecolors='#e377c2')
        # plt.loglog(self.xdata[0], self.ydata[0],'v',color='b')
        # plt.loglog(self.xdata[1], self.ydata[1],'d',color='b')
        # plt.loglog(self.xdata[2], self.ydata[2],'s',color='b')
        # plt.loglog(self.xdata[3], self.ydata[3],'<',color='b')
        # plt.loglog(self.xdata[4], self.ydata[4],'>',color='b')
        # plt.loglog(self.xdata[5], self.ydata[5],'^',color='b')
        # plt.loglog(self.xdata[6], self.ydata[6],'-',lw=1,color='#1f77b4')
        plt.loglog(self.xdata[3], self.ydata[3],'-',lw=1,color='k')

        plt.xlabel("k", fontsize=25)
        plt.ylabel("P(k)", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)  # 修改刻度线线粗细width参数
        ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5)
        ax.legend(labels=[r"$n=2$",r"$n=3$",r"$n=4$","theoretical"], ncol=1, fontsize=20)

        # plt.legend(["p=0","p=0.2","p=0.4","p=0.6","p=0.8","p=1","theoretical"])

        plt.savefig("img.svg",format='svg',dpi=600)  # svg格式
        plt.show()

    # 导入数据
    def importModel(self,N,m1,m2,m,n):  # 节点数(5000)
        print("导入N={},m1={},m2={},m={},n={}的模型".format(N,m1,m2,m,n))
        # BA超网络模型导入
        self.incidenceMatrix = np.genfromtxt(
            self.importpath+"n"+str(N)+"m1"+str(m1)+"m2"+str(m2)+"m"+str(m)+ "_" +str(n)+'version.txt', delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()  # 超边中的节点用元组存储
            for j in range(len(self.incidenceMatrix)):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
            self.scenes["E" + str(i + 1)] = eTuple

    def dataExport(self):
        print("存储...")
        with open(self.dataSave + 'degDist.txt', 'w') as file0:
            print("xData:", file=file0)
            print(self.xdata, file=file0)
            print("yData:", file=file0)
            print(self.ydata, file=file0)
        print()

if __name__ == '__main__':
    Hu = HUHPM()
    # Hu.modelConstruction()
    # Hu.funRun()
    Hu.funRun1()
