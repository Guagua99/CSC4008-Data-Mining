import numpy as np
import matplotlib.pyplot as plt

# Load Dataset from TXT file
def load_data():
  adjacency_matrix = np.loadtxt('D:\\courses\\CSC4008\\Project\\adjacency_matrix.txt', delimiter=',')
  city_coordinate = np.loadtxt('D:\\courses\\CSC4008\\Project\\city_coordinates.txt', delimiter=',')
  return adjacency_matrix, city_coordinate

# Multi-Dimension Scaling Algorithm
def MDS():
  adjacency_matrix, city_coordinate = load_data()
  data_num = city_coordinate.shape[0]
  print('data_num: ', data_num)
  # Proximity Matrix D and D2
  D = np.mat(adjacency_matrix)
  D2 = D ** 2
  # Matrix J
  J = np.eye(data_num) - (np.eye(data_num) / data_num)
  # Matrix B
  B = 0.5 * J * D2 * J
  # Two Eigenvectors with the Largest Eigen Value
  eig_val, eig_vec = np.linalg.eig(B)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(data_num)]
  eig_pairs.sort(reverse=True, key=lambda ele:ele[0])
  eigen_vec_1 = eig_pairs[0][1]
  eigen_vec_2 = eig_pairs[1][1]
  # Matrix X
  EM = np.hstack((eigen_vec_1, eigen_vec_2))
  EigenValueMat = np.eye(2)
  for i in range(2):
    EigenValueMat[i, i] = np.sqrt(eig_pairs[i][0])
  X = EM * EigenValueMat
  print('shape of X: ', X.shape)
  print(X)
  # Verify the Adjacency Distance Matrix
  new_adjacency_matrix = np.zeros(shape=(data_num, data_num))
  for i in range(data_num):
    for j in range(data_num):
      dist = 0
      if i != j:
        dist = np.sqrt(
          abs((X[i, 0] ** 2 - X[j, 0] ** 2))
          +
          abs((X[i, 1] ** 2 - X[j, 1] ** 2))
        )
      new_adjacency_matrix[i, j] = dist
  print('New Adjacency Matrix: \n', new_adjacency_matrix)
  # Plot Original Cities
  plt.scatter(city_coordinate[:, 0], city_coordinate[:, 1], c='red')
  # Plot Projected Cities
  plt.scatter(np.asarray(X[:, 0]).flatten(), np.asarray(X[:, 1]).flatten(), c='blue')
  plt.show()



if __name__ == '__main__':
  MDS()
