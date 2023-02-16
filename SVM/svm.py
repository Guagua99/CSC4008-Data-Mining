import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


random.seed(10)
# generate data
a = np.array([[random.randint(1, 600)/100 for j in range(1, 3)] for i in range(1,101)])
b = np.array([[random.randint(401, 1000)/100 for j in range(1, 3)] for i in range(1,101)])
X = np.vstack((a,b))
y = np.zeros(200, dtype=int)

for i in range(100,200):
    y[i] = np.array(1)
#(200,)


# standardization
standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)  #(200, 2)




def plot_svc_decision_boundary(model, axis):

    y_predict = model.predict(X_standard)

    w = model.coef_[0]
    b = model.intercept_[0]

    # w0 * x0 + w1 * x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1

    y1=-w[0]/w[1] * (-3) - b/w[1]
    y2=-w[0]/w[1] * 3 - b/w[1]
    plt.plot([-3,3],[y1,y2],c='black')
  

    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0]/w[1] * plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1] * plot_x - b/w[1] - 1/w[1]
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color="black",ls='--')
    plt.plot(plot_x[down_index], down_y[down_index], color="black",ls='--')

    # points that alpha_i = C
    plt.scatter(X_standard[(y_predict==1)==(y==0), 0], X_standard[(y_predict==1)==(y==0), 1],s=40,c='r',alpha=0.5,marker='s')


    # points that alpha_i = 0
    for i in range(200):
        x0 = X_standard[i][0]
        y0 = X_standard[i][1]
        y1 = -w[0]/w[1] * x0 - b/w[1] - 1/w[1]
        y2 = -w[0]/w[1] * x0 - b/w[1] + 1/w[1]
        y_min=min(y1,y2)
        y_max=max(y1,y2)
        if ((y_min>y0)or(y_max<y0)):
            plt.scatter(X_standard[i][0],X_standard[i][1],s=40,c='r',alpha=0.5,marker='o')

    # plot all points
    plt.scatter(X_standard[y == 0, 0], X_standard[y == 0, 1],s=20,c='k',alpha=0.5)
    plt.scatter(X_standard[y == 1, 0], X_standard[y == 1, 1],s=20,c='b',alpha=0.5)
    plt.show()
    
svc = LinearSVC(C=0.1)
svc.fit(X_standard, y)

svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, y)


plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])

