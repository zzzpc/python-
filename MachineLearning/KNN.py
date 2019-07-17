import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from collections import  Counter


iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['label']=iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# data = np.array(df.iloc[:100, [0, 1, -1]])
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


data=np.array(df.iloc[:100,[0,1,-1]])
X,y=data[:,:-1] ,data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
class KNN:
    def __init__(self,X_train,y_train,n_neighbor=3,p=2):
        """
               parameter: n_neighbors 临近点个数
               parameter: p 距离度量(默认欧式距离)
        """
        self.n=n_neighbor
        self.p=p
        self.X_train=X_train
        self.y_train=y_train

    def predict(self,X):
        #取出n个点
        knn_list=[]
        for i in range(self.n):
           #计算n个点与测试点之间的距离
            dist=np.linalg.norm(X - self.X_train[i],ord=self.p) #求二阶范式
            knn_list.append((dist,self.y_train[i]))


        for i in range(self.n,len(self.X_train)) :
            max_index=knn_list.index(max(knn_list,key=lambda x: x[0]))#按照x[0]来选择最大值
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p) #求范式(二阶)
            if knn_list[max_index][0] >dist :    #找到最小值
                knn_list[max_index] = (dist,self.y_train[i])


        #统计
        knn=[k[-1] for k in knn_list]
        count_pairs=Counter(knn)
        max_count=sorted(count_pairs.items(),key=lambda x:x[1])[-1][0]
        print(max_count)
        return max_count

    def   score(self,X_test,y_test):
        right_count=0
        n=10
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label == y :
                right_count += 1
        return right_count/len(X_test)

clf = KNN(X_train, y_train)
clf.score(X_test, y_test)
test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))


plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
