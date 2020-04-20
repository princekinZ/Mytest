import math
import numpy as np
from sklearn.cluster import DBSCAN

MIN_DISTANCE = 0.0001
# 高斯核函数
def Gussian_kernel(point1,points,bandwidth):
    left=1 / np.sqrt((2 * math.pi)) # 左边部分
    # 右边部分
    right=np.exp(-0.5*pow(np.linalg.norm(point1-points),2)/(bandwidth*bandwidth))
    return left*right

# 计算高斯核函数之和
def Gussian(point1,points,bandwidth):
    fenzi=0 # 分子，分母分开进行计算
    fenmu=0
    for i in range(points.shape[0]):
        fenzi+=Gussian_kernel(point1,points[i],bandwidth)*points[i] # 分子之和
        fenmu+=Gussian_kernel(point1,points[i],bandwidth) # 分母之和
    return fenzi/fenmu

# 寻找吸引子的函数
def findAttractor(bandwidth,convergence,point,points):
    point1=point # 定义xt 即point1
    point2=point # 定义xt+1 即point2
    while(True): #无限循环，直到满足条件结束循环
        point1=point2 #公式中t=t+1 即让point1=point2
        point2=Gussian(point1,points,bandwidth) # 计算xt+1的值
        # 判断条件xt和xt+1之间的距离是否小于convergence
        if(np.linalg.norm(point2-point1)<convergence):
            return point1

# 密度估计函数，判断这个空间内的点的密度是否大于一个值，来确定这些点是否足够构成一类
def multi(point,points,bandwidth):
    f=0
    # 固定一个区域大小，对点集中的每一个点与point之间的距离进行判断
    for i in range(points.shape[0]):
        f+=Gussian_kernel(point, points[i], bandwidth) # 进行加权，最后求和
    # 得到整个区域中点的分布，再除以区域大小，求得这个区域的密度估计函数
    return f/(points.shape[0] * pow(bandwidth, 4))

# denclue算法
def denclue(points, bandwidth, hyper, convergence):
    A = list() # 定义一个列表存吸引子
    r = dict() # 定义一个字典存吸引子对应的坐标值
    for i in range(points.shape[0]): # 对点集中的每个点判断
        x = findAttractor(bandwidth, convergence, points[i], points) # 求出每个点的吸引子
        # 用密度估计判断这个吸引子，吸引的点在区域内的密度大小。如果太小不足以构成一类就舍弃
        if multi(x, points, bandwidth) > hyper:
            A.append(list(x)) # 像列表A中插入符合条件的吸引子
            strxing = str(x) # 将吸引子的坐标变成字符串
            if strxing not in r.keys(): # 如果字典中没有这个吸引子的坐标值
                a = [] # 定义一个数组
                a.append(points[i]) # 将对应的点全部加入
                r[strxing] = a # 将吸引子的值与这些点对应
            # 有这个值，那就将点直接加入，一个吸引子可以对应很多点
            else:
                a = r[strxing]
                a.append(points[i])
                r[strxing] = a
    E=0.1
    min=2
    b=DBSCAN(eps=E,min_samples=min) # 用DBCSACN进行聚类，密度可达，最大化
    c=b.fit_predict(np.mat(list(A))) # 训练之后的结果
    b_set=set()
    b_dict=dict()
    for i in range(len(c)): # 将训练完之后的类加入集合
        if(c[i]!=-1):
            b_set.add(c[i])
            if str(c[i]) not in b_dict.keys():
                b=[]
                b.append(A[i])
                b_dict[str(c[i])]=b
            else:
                b=b_dict[str(c[i])]
                b.append(A[i])
    print("clusters number",len(b_dict.keys())) # 输出每个类及其中的点

    for i in b_dict.keys(): # 对每一类进行判断，如果吸引子属于该类则加入
        xs=b_dict[i]
        for x in b_dict[i]:
            if str(x) in r.keys():
                for i in r[str(x)]:
                 xs.append(i)
        b_dict[i]=xs

    for i in b_dict.keys():
        print("cluster",i)
        print("points in cluster",b_dict[i])

if __name__ == "__main__":
    D = np.loadtxt('iris.txt', delimiter=',', usecols=(0, 1, 2, 3))
    m, n = np.shape(D)
    denclue(D,0.27,4.0,0.0001)