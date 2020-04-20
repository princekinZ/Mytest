# 导入库
import numpy as np
import matplotlib.pyplot as plt

# 导入magic04,txt文件
dataM = open('magic04.txt')
# 读取所有字符串
List_row = dataM.readlines()
List_result = []
for i in range(len(List_row)):
    # 用“，”进行分割
    column_list = List_row[i].strip().split(",")
    # 将最后一列的g去除
    column_list.pop()
    # 将前10列数据加入list_result中
    List_result.append(column_list)
# 转化为numpy数组npArray
npArray=np.array(List_result)
# 将数组npArray转换为浮点类型
npArray=npArray.astype(float)

# （1）计算多元均值向量
meanVector=npArray.mean(axis=0)
print("多元均值向量为：")
print(meanVector)

# 求中心矩阵center
center=npArray-np.ones((npArray.shape[0],1),dtype=float)*meanVector
# (2)计算样本协方差矩阵，通过中心数据矩阵列之间的内积
innerP=np.dot(center.T,center)
cov1=innerP/npArray.shape[0]
print("以内积方式求协方差矩阵：")
print(cov1)

# (3)计算样本协方差矩阵，通过中心化数据点之间的外积
cov2=np.cov(npArray.T)
print("以外积方式求协方差矩阵：")
print(cov2)

# （4）计算属性1和属性2的夹角
corrC=np.corrcoef(center.T[0],center.T[1]) # 调用np.corrcoef()计算两属性的余弦值矩阵
print("向量夹角为：")
print(corrC[0][1])
# （4）绘制两属性的散点图
figu = plt.figure()
ax1 = figu.add_subplot(111) # 参数111：画布分割为1行1列，图像在第一块
ax1.set_title('Scatter Diagram')
plt.scatter(center.T[0],center.T[1],c='green',alpha=0.6)
# 设置轴标签
plt.xlabel('attrValue1')
plt.ylabel('attrValue2')
plt.show()

# (5)假设属性1正态分布，画出其概率密度函数(概率密度函数公式：np.exp())
# 第一列均值，调用np.mean（）
meanVal=np.mean(npArray,axis=0)[0]
# 第一列方差，调用np.var（）
var=np.var(npArray.T[0])
figu = plt.figure()
ax1 = figu.add_subplot(111)
# 正态分布公式
x = np.linspace(meanVal - 3*var, meanVal + 3*var, 50)
y_var = np.exp(-(x - meanVal) ** 2 /(2* var **2))/(np.sqrt(2*np.pi)*var)
plt.plot(x, y_var, "green", linewidth=2,alpha=0.6)
plt.title("Normal Quantie Of attrValue1")
plt.xlabel("centAttrValue1")
plt.ylabel("probability")
plt.show()

# (6)求出方差最小和最大的属性
# 求每一列的方差
list=[]
for i in range(len(npArray[0])):
    list.append(np.var(npArray.T[i]))
maxIndex=list.index(max(list))
minIndex=list.index(min(list))
# 返回列
print("第",maxIndex+1,"列的方差最大")
print("第",minIndex+1,"列的方差最小")

# （7）求协方差最大和最小的两个属性组合
# 求矩阵两列协方差
cov={}
for i in range(9):
    for j in range(i+1,10):
        str1=str(i+1)+'和'+str(j+1)
        cov[str1]= np.cov(npArray.T[i],npArray.T[j])[0][1]
print("协方差最小的两列属性组合：")
print(min(cov, key=cov.get))
print("协方差最大的两列属性组合：")
print(max(cov, key=cov.get))