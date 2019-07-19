from math import exp
import  numpy as np
import  matplotlib.pyplot as plt
import  pandas as pd

from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split


def create_data() :
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label']=iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data=np.array(df.iloc[:100,[0,1,-1]])
    return data[:,:2] ,data[:,-1]

X,y=create_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

class LogisticRegressionClassifier:
    def __init__(self,max_iter=200,learning_rate=0.01):
        self.max_iter=max_iter #最大迭代次数
        self.learning_rate=learning_rate

    def sigmoid(self,x):
        return  1/(1+exp(-x))

    def data_matrix(self, X):   #构建数据集矩阵
        data_mat = []
        for d in X:
            data_mat.append([1.0, d[0],d[1]])  #这里也可以用*d来表示(d[0],d[1])
        return data_mat

    def fit(self,X,y):
        data_mat=self.data_matrix(X)
        self.weights=np.zeros((len(data_mat[0]),1),dtype=np.float32)
        for iter in range(self.max_iter):#开始迭代
            for i in range(len(X)) :
                result=self.sigmoid(np.dot(data_mat[i],self.weights))  #输入向量与w向量的点积
                error=y[i]-result   #错误度
                self.weights+=self.learning_rate*error*np.transpose([data_mat[i]])  #对权重w更新
            #    print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
              #      self.learning_rate, self.max_iter))

    def score(self,X_test,y_test):
        right=0
        X_test=self.data_matrix(X_test)
        for x,y in zip(X_test,y_test):
            result=np.dot(x,self.weights)
            if(result>0 and y==1) or (result<0 and y==0):
                right+=1
        return right/len(X_test)

lr_clf =LogisticRegressionClassifier()
lr_clf.fit(X_train, y_train)
x_ponits = np.arange(4, 8)
print(lr_clf.weights)
y_ = -(lr_clf.weights[1]*x_ponits + lr_clf.weights[0])/lr_clf.weights[2] #貌似与感知机中最后计算的线性函数过程相似
plt.plot(x_ponits, y_)

#lr_clf.show_graph()
plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.show()
