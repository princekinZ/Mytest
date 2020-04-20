from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

iris = datasets.load_iris() # 获取数据
feature = iris.data # 特征数据
target = iris.target # 分类数据

# 随机划分训练集和测试集
# train_test_split()：用于随机划分训练集和测试集
# test_size：表示划分到测试集数据占全部数据的百分比
# random_state：随机数的种子，表示乱序程度
feature_train,feature_test,target_train,target_test = train_test_split(feature,target,test_size=0.3,random_state=1)

# 使用决策树对iris进行模型训练及预测
default_model = DecisionTreeClassifier() # 所有参数设置为默认状态
default_model.fit(feature_train,target_train) # 使用训练集训练模型
target_predict = default_model.predict(feature_test) # 使用模型对测试集进行预测

# 获取结果报告
print('accracy score:',default_model.score(feature_test,target_test))
print(classification_report(target_predict,target_test,target_names=['iris-setosa', 'iris-versicolor', 'irisvirginica']))

# 用graphviz进行决策树图的输出
with open("decision-tree-iris.dot","w") as f:
    f = tree.export_graphviz(default_model, out_file=f,feature_names=iris.feature_names,class_names=iris.target_names)
# 进入cmd，切换到目标文件目录
# 输入指令  dot -Tpdf decision-tree-iris.dot -o decision-tree-iris.pdf

