# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot
from sklearn.datasets import  load_iris
import  pandas as pd
import  random

class  K_means(object):
        def __init__(self,k=2,tolerance=0.0001,max_iter=300):
            self.k_=k
            self.tolerance_=tolerance
            self.max_iter_=max_iter

        def fit(self,data):
            self.centers_={}
            for i in range(self.k_) :
                self.centers_[i]=data[i]

            for i in range(self.max_iter_):
                self.clf_={}
                for i in range(self.k_):
                    self.clf_[i]=[]  #分成k个组
                # print("质点:",self.centers_)
                for feature in data :
                    distances=[]
                    for center in self.centers_:
                        #欧氏距离(第二范式)  求数据集中每一个点到中心点的距离
                        distances.append(np.linalg.norm(feature-self.centers_[center]))
                    #选择最小的距离下标作为分类中心
                    classification=distances.index(min(distances))

                    self.clf_[classification].append(feature)
                #保存为更新之前的中心点
                prev_centers=dict(self.centers_)#经列表转换成字典形式
                for c in self.clf_:
                    #更新每个中心类的坐标，取平均值
                    self.centers_[c]=np.average(self.clf_[c],axis=0)

                #中心点是否在误差范围内
                optimized=True
                for center in self.centers_ :
                    org_centers=prev_centers[center]
                    cur_centers=self.centers_[center]
                    if np.sum((cur_centers-org_centers)/org_centers *100)>self.tolerance_ :
                        optimized=False
                if optimized :
                    break


        def  predict(self,p_data):
            distance=[np.linalg.norm(p_data-self.centers_[center]) for center in self.centers_]
            index=distance.index(min(distance))
            return index


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    data = np.array(df.iloc[:100, [0, 1]])
    X = data[:, :]
    x=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
    print(x)
    k_means=K_means(k=4)
    k_means.fit(X)
    print(k_means.centers_)
    color=['red','blue','yellow','black','orange']
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0],k_means.centers_[center][1],marker='*',s=180,color=color[center])

    for cat in k_means.clf_:
        for point in k_means.clf_[cat] :
            pyplot.scatter(point[0],point[1],c=color[cat])

    predict=[[6,2],[4,5]]
    for feature in predict :
        car=k_means.predict(predict)
        pyplot.scatter(feature[0],feature[1],color='pink')

    pyplot.show()
