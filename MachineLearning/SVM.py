import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from collections import  Counter
from sklearn.neighbors import  KNeighborsClassifier


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

class SVM :
    def __init__(self,max_iter=100,kernel='linear'):
        self.max_iter=max_iter
        self._kernel=kernel

    def init_args(self, features, label):
        self.m, self.n = features.shape
        self.X = features
        self.y = label
        self.b = 0.0

        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0

    def _KKT(self,i):
         y_g=self._g(i)*self.y[i]   #KKT条件
         if self.alpha[i] ==0 :
             return y_g >=1
         elif 0<self.alpha[i]<self.C :
             return  y_g == 1
         else :
             return y_g<=1

    # g(x)预测值，输入xi（X[i]）  分类决策函数(sign  1-n的求和)
    def _g(self,i):    #对应于公式7.104   g(x)表示对输入xi的预测值
        r=self.b
        for j in range(self.m) :
            r += self.alpha[j] * self.y[j] * self.kernel(self.X[i], self.X[j])
        return r

    #核函数
    def kernel(self,x1,x2):
        if self._kernel =='linear' :
            return sum([x1[k]*x2[k] for k in range(self.n)])
        elif self.kernel == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.n)])+1)**2
        return  0

    # E（x）为g(x)对输入x的预测值和y的差   对应公式7-105
    def _E(self,i):
        return self._g(i) - self.y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list=[i for i in range(self.m) if 0<self.alpha[i]<self.C]
        #否则遍历整个训练集
        non_satisfy_list=[i for i in range(self.m) if i not in index_list ]
        index_list.extend(non_satisfy_list)
        #append 与extend 的区别是  append是将整个添加的内容以对象的形式存进list中，而extend是一序列的形式填充进序列
        for i in index_list:
            if self._KKT(i) :
                continue
            E1=self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j=min(range(self.m),key=lambda x:self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    def _compare(self,_alpha,L,H): #满足alpha大于0小于C   对应于公式 7-108   亦即经过剪辑后的解
        if _alpha >H :
            return  H
        elif _alpha <L :
            return L
        else:
            return _alpha
    def fit(self,feature,labels):  #smo算法
        self.init_args(feature,labels)
        for t in range(self.max_iter):
            i1,i2 =self._init_alpha()  #返回初始化后的i，j

            if self.y[i1] ==self.y[i2]:   #判断y1是否等于y2
                L=max(0,self.alpha[i1]+self.alpha[i2]-self.C)
                H=min(self.C,self.alpha[i1]+self.alpha[i2])
            else:
                L=max(0,self.alpha[i2]-self.alpha[i1])
                H=min(self.C,self.C+self.alpha[i2]-self.alpha[i1])

            E1=self.E[i1]
            E2=self.E[i2]
            #eta=K11+K22-2K12 对应于公式7-107
            eta=self.kernel(self.X[i1],self.X[i1])+self.kernel(self.X[i2],self.X[i2])-2*self.kernel(self.X[i1],self.X[i2])
            if eta <=0 :
                continue

            #对应于公式7-106  表示沿着约束方向未经剪辑时的解
            alpha2_new_unc=self.alpha[i2] +self.y[i2]*(E1-E2)/eta
            alpha2_new=self._compare(alpha2_new_unc,L,H)  #计算经过剪辑后的解
            #通过计算出来的alpha2来计算新的alpha1  对应公式7.109
            alpha1_new=self.alpha[i1]+self.y[i1]*self.y[i2]*(self.alpha[i2]-alpha2_new)
            #对应公式7-115  在每次完成两个变量的优化后，重新计算阈值b
            #此时是new_alpha1满足条件  0<alpha1<C   对应公式 7-115
            b1_new = -E1 - self.y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                    alpha1_new - self.alpha[i1]) - self.y[i2] * self.kernel(
                self.X[i2],
                self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            # 此时是new_alpha2满足条件  0<alpha2<C  对应公式 7-116
            b2_new = -E2 - self.y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                    alpha1_new - self.alpha[i1]) - self.y[i2] * self.kernel(
                self.X[i2],
                self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            #检测哪个alpha 满足条件 及
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

                # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

    def predict(self, data):
            r = self.b
            for i in range(self.m):
                r += self.alpha[i] * self.y[i] * self.kernel(data, self.X[i])

            return 1 if r > 0 else -1

    def score(self, X_test, y_test):
            right_count = 0
            for i in range(len(X_test)):
                result = self.predict(X_test[i])
                if result == y_test[i]:
                    right_count += 1
            return right_count / len(X_test)

    def _weight(self):
            # linear model
            yx = self.y.reshape(-1, 1) * self.X
            self.w = np.dot(yx.T, self.alpha)
            return self.w
    def _b(self):
        return self.b

clf=SVM()
clf.fit(X,y)
print(clf._weight())
plt.scatter(X[:50,0],X[:50,1],c='pink',label='0')
plt.scatter(X[51:,0],X[51:,1],c='blue',label='0')
plt.show()
