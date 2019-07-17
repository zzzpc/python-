# coding:utf-8

import  pandas as pd
import numpy as np
from sklearn.datasets import  load_iris
import matplotlib.pyplot as plt


#加载数据集
iris=load_iris()
df=pd.DataFrame(iris.data ,columns=iris.feature_names)

df['label']=iris.target

df.columns=['sepal length','sepal width','petal length ','petal width','label']
df.label.value_counts()    #label这一列各种取值分布情况

"""
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'],label='0',color='yellow')
plt.scatter(df[100:150]['sepal length'],df[100:150]['sepal width'],label='1',color='pink')
#plt.scatter(df[101:150]['sepal length'],df[101:150]['sepal width'],label='2',color='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
#plt.legend()
"""

data1=np.array(df.iloc[:50,[0,1,-1]])
data2=np.array(df.iloc[51:100,[0,1,-1]])
data=np.vstack((data1,data2)) #两个数组的垂直合并，不改变原有的形式
data=np.array(data)
print(data)
#data是一个列表每个元素包含df中的前两列和最后一列元素
X, y = data[:,: -1], data[:,-1]
#X为选取所有子元素的第0列至最后一列之间的元素，y为所有元素的最后一列
y=np.array([1 if i == 1 else  -1  for i in y])  #将其分为正负两类


# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data  #亦即学习率   对应η

    def sign(self,x,w,b):
        y=np.dot(x,w)+b #符号函数
        return y
    #随机梯度下降法
    def  fit(self,X_train,y_train):
        count=99999  #设置步长

        while count !=0:
            wrong_count = 0
            for d in  range(len(X_train)) :
                X=X_train[d]
                y=y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0: #当结果小于0是证明误分类点出现
                    self.w=self.w+self.l_rate*np.dot(y,X)  #这里y与X是点积(注意，y与X是向量)
                    self.b=self.b+self.l_rate*y
                    wrong_count+=1
            if wrong_count == 0:
                break
            count=count-1
            if count == 0 :
                print("拟合失败！！！")

        return '梯度下降完成，找到最优的w,b'

    def score(self):
        pass

perception=Model()
perception.fit(X,y)
print(perception.w)
print(perception.b)
x_points=np.linspace(4,8,50)
y_=-(np.dot(perception.w[0],x_points)+perception.b)/perception.w[1]  #perception.w[1]为w[0]的范式
plt.plot(x_points,y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[51:, 0], data[51:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
