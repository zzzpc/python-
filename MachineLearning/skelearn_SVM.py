
import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import  train_test_split
from sklearn.datasets import  make_circles
from sklearn.preprocessing import PolynomialFeatures

X,y=make_circles(n_samples=200,noise=0.05,random_state=42)  #构建一个圆形散点数据集，noise为噪声 factor：控制内外圈的接近程度，越大越接近，上限为1

def plot_data(X,y,axis):
    plt.plot(X[:,0][y==0] ,X[:,1][y==0],"bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axis)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_l$")
    plt.ylabel(r"$x_2$")

def  plot_predict(clf,axis):
    x0s=np.linspace(axis[0],axis[1],100)
    x1s=np.linspace(axis[2],axis[3],100)
    x0,x1=np.meshgrid(x0s,x1s)  #将x0s,x1s变换成坐标矩阵

    print(x0.ravel())
    print('--------------0')
    print(x1.ravel())
    X=np.c_[x0.ravel(),x1.ravel()]  #按行连接两个矩阵，把两个矩阵左右相加   r_(按列连接矩阵，矩阵上下相加，要求列数相等)
    # ravel函数 对矩阵做扁平化处理
    print(X)
    y_pred=clf.predict(X).reshape(x0.shape)
    y_decision=clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5)  #绘制等高线  cmap表示渐变标准，alpha表示 透明度
    plt.contour(x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2)

polynomial_svm_clf = Pipeline([ ("poly_featutres", PolynomialFeatures(degree=3)),
                                ("scaler", StandardScaler()),
                                ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42)  )
                            ])

polynomial_svm_clf.fit(X,y)
plot_data(X,y,[-1.5,2.5,-1,1.5])
plot_predict( polynomial_svm_clf, [-1.5, 2.5, -1, 1.5] )
plt.show()