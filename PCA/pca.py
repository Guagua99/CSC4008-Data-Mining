import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def pca(X,n):  
    new_data=X-np.mean(X,axis=0)  # moving the center
    cov_matrix=np.cov(X,rowvar=0)    # calaulate covariance matrix      
    eigen_vals,eigen_vects=np.linalg.eig(np.mat(cov_matrix)) # calculate eigenvalues and eigen vectors  
    eigen_val_sorted=np.argsort(eigen_vals)            # sort eigenvalues
    n_eigval_indice=eigen_val_sorted[-1:-(n+1):-1]   # get the top n eigenvalue's indices  
    n_eigvect=eigen_vects[:,n_eigval_indice]        # get the  top n eigenvectors
    pca_result=new_data*n_eigvect               # get the low dimension data

    return pca_result

def get_data():
    digits = datasets.load_digits()
    X = digits.images
    Y = digits.target
    X_0 = X[(Y == 0)]
    Y_0 = [0 for i in range(X_0.shape[0])]
    X_1 = X[(Y == 1)]
    Y_1 = [1 for i in range(X_1.shape[0])]

    X_new = np.concatenate((X_0, X_1), axis=0).reshape((360, 64)).T
    Y_new = np.concatenate((Y_0, Y_1), axis=0)
    # print(X_new.shape)
    # print(X_new)
    return X_new, Y_new



if __name__ == '__main__':
    X, Y = get_data()
    X = X.transpose()
    pca_result = pca(X,2)
    
    x0=np.ravel(pca_result[:,0])
    x0=np.real(x0)

    x1=np.ravel(pca_result[:,1])
    x1=np.real(x1)

    X_0_PC1 = x0[(Y == 0)]
    X_0_PC2 = x1[(Y == 0)]
    plt.scatter(X_0_PC1, X_0_PC2, marker='^', c="r")
    
    X_1_PC1 = x0[(Y == 1)]
    X_1_PC2 = x1[(Y == 1)]
    plt.scatter(X_1_PC1, X_1_PC2, marker='o', c="b")

    plt.show()

