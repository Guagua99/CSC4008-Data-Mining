import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
def PCADimReduction(X,dim):
    """
    :param X: input data, X=[x1,x2,...,xn].T, xi is column vector
    :param dim: low dimension embedding
    :return: A -data in lower dimension space.
             x0 -mean value of X
             sort_evecs -eigen-vectors, the Gaussian structure of higher dimension data, also the projection function
             sort-evals -the eigen value corresponding to the eigen-vectors
    """
    # de-duplicate
    #X = np.mat(np.array(list(set([tuple(t) for t in X]))))
    X=np.mat(X)
    X = X.T

    # find means for samples, centerilized, and get scatter matrix
    DIM, num = np.shape(X)
    x0=np.average(X,axis=1)
    H=(np.eye(num)-1/num*np.ones(num))/num
    Scatter=(X)*H*(X.T)
    Scatter[np.isnan(Scatter)] = 0
    print(1)
    # get eigen values and vectors (all of the vector is column vector)
    evals,evecs=np.linalg.eig(Scatter)          # get eigen-value, eigen-vector
    sort_indices=np.argsort(-evals)             # sort the eigen-value from big to small
    sort_evals=evals[sort_indices[:dim]]        # sort the first 'dim's big eigen-value and eigen-vector
    sort_evecs=evecs[:,sort_indices[:dim]]
    A=(sort_evecs.T)*X                          # compute the dimensionality reduction matrix

    return A,x0,sort_evecs,sort_evals

pictures=[]
size=np.array([200,150])
for i in range(16):
    if i <9:
        im=misc.imread("0"+str(i+1)+".jpg")             # open pictures
    else:
        im=misc.imread(str(i+1)+".jpg")                 # open pictures
    ig=np.dot(im[..., :3], [0.299, 0.587, 0.114])       # RGB to Gray
    im=misc.imresize(ig,(size[0],size[1]),'bilinear')   # transfer to fixed size
    misc.imsave("gray_0" + str(i + 1) + ".jpg", im)     # save the transferred pictures
    im=im.reshape([size[0]*size[1],1])                  # flat 2D data to 1D
    pictures.append(im)                                 # add it to the array

pictures=(np.array(pictures)/255).reshape(16,size[0]*size[1])

A,x0,sort_evecs,sort_evals=PCADimReduction(pictures,100)

R=sort_evecs*A+x0
R.reshape([16,size[0],size[1]])
plt.imshow(R[1,...])
plt.show()

print(1)