# 导入库
import numpy as np

# 读取iris.txt文件
list_result = []
with open('iris.txt','r') as f:
    for line in f:
        # 用“，”进行分割
        line=list(map(str,line.split(',')))
        # 将最后一列得种类属性去掉
        list_result.append(list(map(float,line[:4])))
# 转化为numpy数组npArray
npArray=np.array(list_result)
# 将数组npArray转换为浮点类型
npArray=npArray.astype(float)

# (1)通过核函数，计算经过中心化和归一化的核矩阵K
# 初始化K
K=np.empty((len(npArray),len(npArray)))
# 双重循环遍历，计算每一个K元素的值
for h in range(len(npArray)):
    for j in range(len(npArray)):
        K[h][j]=np.dot(npArray[h],npArray[j])
K=K**2 # 平方得齐次二次核
# 根据公式完成核矩阵中心化
I=np.eye(len(npArray),M=None,k=0) # 单位矩阵 对角线全为1
center=I-1/len(npArray)*np.ones((len(npArray),len(npArray))) # 中心化
normalizeK=np.dot(np.dot(center,K),center) # 归一化
W=normalizeK*I # 中心化矩阵和单位矩阵的外积
for i in range(len(npArray)):
    W[i][i]=1/np.sqrt(W[i][i])
nuK=np.dot(np.dot(W,normalizeK),W)
print("所求核矩阵为：")
print(nuK)

# (2)将四维向量升到10维向量
Raise=np.zeros((len(npArray),10))
for i in range (len(npArray)):
    for j in range(4):
        Raise[i][j] = npArray[i][j]*npArray[i][j] #前四个属性为平方
    Raise[i][4] = np.sqrt(2) * npArray[i][0] * npArray[i][1]
    Raise[i][5] = np.sqrt(2) * npArray[i][0] * npArray[i][2]
    Raise[i][6] = np.sqrt(2) * npArray[i][0] * npArray[i][3]
    Raise[i][7] = np.sqrt(2) * npArray[i][1] * npArray[i][2]
    Raise[i][8] = np.sqrt(2) * npArray[i][1] * npArray[i][3]
    Raise[i][9] = np.sqrt(2) * npArray[i][2] * npArray[i][3]
print("映射结果为：")
print(Raise)

# (3)将Raise中心化和标准化，然后两两求点积
# 中心化
meanVal=np.mean(Raise,axis=0) # 求每一列的均值
Raise=Raise-meanVal  # 均值向量维度为二,直接相减
K3=np.zeros((len(npArray),len(npArray)))
# 标准化
length=[]
for i in range (len(npArray)): # 计算每行的模
    length.append(np.sqrt(sum(ei*ei for ei in Raise[i])))
    Raise[i]=Raise[i]/length[i]
# 求两两点积
K3=np.zeros((len(npArray),len(npArray)))
for i in range(len(npArray)):
    for j in range(len(npArray)):
        K3[i][j]=np.dot(Raise[i],Raise[j])
print("特征空间中经中心化和标准化的核矩阵：")
print(K3)
