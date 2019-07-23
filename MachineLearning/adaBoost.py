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


class AdaBoost:
      def __init__(self,n_estimators=50,learning_rate=1.0):
          self.clf_num=n_estimators
          self.learning_rate=learning_rate

      def init_args(self,dataset,labels):
          self.X=dataset
          self.y=labels
          self.M,self.N=dataset.shape

          #弱分类器集合
          self.clf_sets=[]

          #初始化weights  w发代表初始各数据子集的权重
          self.weights=[1.0/self.M]*self.M

          #G(x)系数 alpha  G(x)表示基本分类器
          self.alpha=[]

      def _G(self,feature, labels,weights):
          m=len(feature)
          error=100000.0
          best_v=0.0
          #单维features
          feature_min=min(feature)
          feature_max=max(feature)
          n_step=(feature_max-feature_min+self.learning_rate)
            # print('n_step {}'.format(n_step))
          direct,compare_array =None, None
          for i in range(1, int(n_step)) :
              v= feature_min + self.learning_rate*i
              if v not in feature :
                  #误分类计算
                  compare_array_positive=np.array([1 if feature[k]>v else -1 for k in range(m)])
                  weight_error_positive=sum([weights[k] for k in range(m) if compare_array_positive[k]!= labels[k]])

                  compare_array_nagetive = np.array([-1 if feature[k] > v else  1 for k in range(m)])
                  weight_error_nagetive= sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])

                  if weight_error_positive <weight_error_nagetive :
                      weight_error=weight_error_positive
                      _compare_array=compare_array_positive
                      direct='positive'

                  else:
                      weight_error = weight_error_nagetive
                      _compare_array = compare_array_nagetive
                      direct = 'nagetive'

                  #print('v:{} error:{}'.format(v,weight_error))
                  if weight_error<error :
                      error =weight_error
                      compare_array=_compare_array
                      best_v=v
          return  best_v ,direct ,error ,compare_array

      def  _alpha(self,error): #计算alpha  error为分类误差率
          return  0.5*np.log((1-error)/error)   #计算Gm(x)的系数， 亦即公式8.2

      def  _Z(self,weights,a,clf):#对于公式8.5
          return sum([weights[i]*np.exp(-1*a*self.y[i] * clf[i]) for i in range(self.M)])

      #更新权值
      def _w(self,a,clf,Z):
          for i in range(self.M) :
              self.weights[i]=self.weights[i]*np.exp(-1*a*self.y[i]*clf[i])/Z #对于公式8.4

      #整合G(x)  线性组合
      def _f(self,alpha, clf_sets):
          pass



      def G(self, x, v, direct):#构建基本分类器(二分类，比较阈值即可)
          if direct == 'positive':
              return 1 if x > v else -1
          else:
              return -1 if x > v else 1

      def  fit(self,X,y):
            self.init_args(X,y)
            for epoch  in  range(self.clf_num) :
                #best_clf_error表示分类误差率，best_v代表最佳阈值
                best_clf_error,best_v,clf_result=100000,None,None
                #根据特征维度，选择误差最小的
                for j in range(self.N):
                    feature=self.X[:,j]
                    #分类阈值，分类误差，分类结果
                    v,direct,error,compare_array=self._G(
                        feature,self.y,self.weights
                    )

                    if error <best_clf_error :
                        best_clf_error=error
                        best_v=v
                        final_direct =direct
                        clf_result=compare_array
                        axis=j

                    if best_clf_error == 0 :
                      break
                a=self._alpha(best_clf_error)
                self.alpha.append(a)
                #记录分类器
                self.clf_sets.append((axis,best_v,final_direct))
                Z=self._Z(self.weights,a,clf_result) #记录新的规范因子
                self._w(a,clf_result,Z) #利用新计算出的规范因子来更新w
      def  predict(self,feature):
          result=0.0
          for i in range(len(self.clf_sets)) :
              axis,clf_v,direct=self.clf_sets[i]
              f_input=feature[axis]
              result+=self.alpha[i] *  self.G(f_input,clf_v,direct)#整合分类器 对应公式8.6

          return  1 if  result>0 else -1

      def score(self,X_test,y_test):
          right_count=0
          for i in range(len(X_test)):
              feature=X_test[i]
              if self.predict(feature) ==y_test[i] :
                  right_count+=1
          return right_count/len(X_test)

"""
X = np.arange(10).reshape(10, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
clf = AdaBoost(n_estimators=3, learning_rate=0.5)
clf.fit(X, y)
"""
X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
clf = AdaBoost(n_estimators=10, learning_rate=0.2)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
