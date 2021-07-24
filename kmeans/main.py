import random
import pandas as pd
import math
import matplotlib.pyplot as plt

#使用pandas导入csv数据
#数据需要带有表头
raw_dataset = pd.read_csv('./data/iris.csv', delimiter=',') #在这里导入数据集
dic = {} #建立字典 解析文本值 转换为相应label
dataset = [data for data in raw_dataset.values]
#给每个文本数据添加标签
for x,features in enumerate(dataset):
    for i,value in enumerate(features):
        if type(value) == str:
            dic[value] = len(dic)
for x,features in enumerate(dataset):
    for i,value in enumerate(features):
        if type(value) == str:
            dataset[x][i] = dic[value]
print("文本化标签数据：{}".format(dic)) #展示不同分类名标记数字
print("参数：{}".format(len(dataset[0])))
print("数据量：{}".format(len(dataset)))
raw_dataset.describe()
#展示原始数据统计
#将数据化成无文本数据

class Cluster(object):
    
    #   定义簇类
    #参数类型：Points 一个离散点的列表
            #Centroid 代表中心质点的坐标

    def __init__(self,points):
        self.points = points
        self.dimension = len(points[0])
        self.centroid = self.calculateCentroid()
        
    #默认选取第一第二个参数可视化展示
    def getXValues(self):
        xValues = []
        for points in self.points:
            x = points[0]
            xValues.append(x)

        return xValues

    def getYValues(self):
        YValues = []
        for points in self.points:
            y = points[1]
            YValues.append(y)

        return YValues

    def calculateCentroid(self):
        #寻找中心点的坐标
        
        if (len(self.points) != 0):
            numberOfPoints = len(self.points)
            coordinates = []
            for points in self.points:
                coordinates.append(points)
                unZippedPoints = zip(*coordinates)
                centroid = [math.fsum(point)/numberOfPoints for point in unZippedPoints]

            return centroid
        else:
            return [0 for i in range(self.dimension)]


    def update(self, points):
        """
           计算簇移动位置距离
        """
        oldCentroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        moved = calculateDistance(oldCentroid, self.centroid)

        return  moved

def sse(clusterArray):
    #参数：clusterArray: 簇（数组）
    #返回：簇内误差平方和
    
    sumCounter = 0
    for cluster in clusterArray:
        for point in cluster.points:
            distance = calculateDistance(point,cluster.centroid)
            sqauredDist = distance**2
            sumCounter = sumCounter + sqauredDist

    return sumCounter

def kmeans(points,dimensions,stoppingCondition):
    initialValues = random.sample(points, dimensions)
    clusters = [Cluster([points]) for points in initialValues]
    counter = 0
    while True:
        listOfPointsinCluster = [[] for _ in clusters]
        clusterCount = len(clusters)
        counter = counter + 1
        for point in points:
            smallestDist = calculateDistance(point, clusters[0].centroid)
            index = 0

            for i in range(clusterCount-1):
                distance = calculateDistance(point,clusters[i+1].centroid)

                if distance < smallestDist:
                    smallestDist = distance
                    index = i+1
            listOfPointsinCluster[index].append(point)

        jump = 0.0

        for i in range(clusterCount):
            moved = clusters[i].update(listOfPointsinCluster[i])

            jump = max(jump,moved)

        if jump < stoppingCondition:
            break
    return clusters


#运行kmeans至少10次，以获得最佳的随机点和SSE

def runKmeans(dataSets, k, stoppingCriteria):
    listOfClusters = []

    counterTest = 0
    while counterTest <= 10:
        cluster = kmeans(dataSets,k,stoppingCriteria)
        listOfClusters.append(cluster)
        counterTest = counterTest + 1

    smallestSSE = 100000
    counter = 0
    for i in range(0,len(listOfClusters)):
        SSE = sse(listOfClusters[i])

        if SSE < smallestSSE:
            smallestSSE = SSE
            counter = i

    return listOfClusters[counter]

#计算两点之间的欧几里德距离
def calculateDistance(list1, list2):
    counter = 0
    for i in range(len(list1)):
        value = list1[i] - list2[i]
        valueSquare = math.pow(value,2)
        counter = counter + valueSquare

    euclid = math.sqrt(counter)

    return euclid

def main():
    
    listOfData = dataset

    #计算从1到20点簇内均分误差（SSE）
    sseArray = []
    for i in range(1, 21):
        clusters = runKmeans(listOfData,i,0.2)
        sseVal = sse(clusters)
        sseArray.append(sseVal)
    kValues = []
    for i in range(1,21):
        kValues.append(i)

    #绘制K值与簇内均分误差（SSE）的图像
    plt.figure(1)
    plt.ylabel("SSE")
    plt.xlabel("K")
    plt.title("Best K Value")
    plt.plot(kValues, sseArray, "-ro", markersize=3)
    plt.show()

    #默认选取超参数K值为5
    #可自行调整超参数值
    actualClusters = kmeans(listOfData,2,0.2)
    sseVal = sse(actualClusters)
    print("簇内均分误差SSE为{}".format(str(sseVal)))
    clusterpoint = []
    for cluster in actualClusters:
        x = cluster.points
        clusterpoint.append([cluster.centroid[0], cluster.centroid[1]])
        print("以坐标{}的点为中心的点有{}个".format(str(cluster.centroid),str(len(x))))
        
    colors = ['bo','ro','go','yo','ko','y','r']
    plt.figure(2)
    plt.ylabel("Y Values")
    plt.xlabel("X Values")
    plt.axis([0, 10, 0, 10])
    plt.title("Clusters")
    for i in range(len(actualClusters)):
        testCluster = actualClusters[i]
        plt.plot(testCluster.getXValues(), testCluster.getYValues(), colors[i],[i[0] for i in clusterpoint],[i[1] for i in clusterpoint],'rx' )
    plt.show()

if __name__=='__main__':
    main()