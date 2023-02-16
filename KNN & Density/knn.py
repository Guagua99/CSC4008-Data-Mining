import numpy as np
import scipy.io as sio

# global variables
BASEPATH_ATNT = "./ATNT face"
BASEPATH_BINALPHA = "./Binalpha handwritten"
TRAINING_DATA_SUBPATH = "/trainX.mat"
TRAINING_LABEL_SUBPATH = "/trainY.mat"
TEST_DATA_SUBPATH = "/testX.mat"
TEST_LABEL_SUBPATH = "/testY.mat"
VALIDATION_DATA_RATIO = 0.15      
K_MIN = 1                         
K_MAX = 20                       

# combine the data and the lable into one table
def get_combined_set(dataset_filepath, label_filepath):
  datasetMat = sio.loadmat(dataset_filepath)
  labelMat = sio.loadmat(label_filepath)
  dataset = datasetMat[list(datasetMat.keys())[-1]]
  labelset = labelMat[list(labelMat.keys())[-1]]
  combined_set = np.r_[labelset, dataset]
  return combined_set

# get the training set and validation set
def get_train_validate_set(dataset, validation_data_ratio = 0.2):
  total_data_num = dataset.shape[1]
  validationDataNum = int(total_data_num * validation_data_ratio)
  row_rand_array = np.arange(total_data_num)
  np.random.shuffle(row_rand_array)
  training_set = dataset[:, row_rand_array[validationDataNum:]]
  validation_set = dataset[:, row_rand_array[0:validationDataNum]]
  training_label_set = training_set[0]
  validation_label_set = validation_set[0]
  training_data_set = np.delete(training_set, 0, axis=0)
  validation_data_set = np.delete(validation_set, 0, axis=0)
  return training_data_set, training_label_set, validation_data_set, validation_label_set

# KNN classifier
def knn_classify(input_data, training_data_set, training_label_set, k):
  training_data_set_size = training_data_set.shape[1]
  input_data = input_data.reshape(input_data.shape[0], -1) #644*1
  diff_mat = np.tile(input_data, (1, training_data_set_size)) - training_data_set #644*272
  sq_diff_mat = diff_mat ** 2
  sq_distances = sq_diff_mat.sum(axis=0)
  distances = sq_distances ** 0.5  #欧式距离
  sorted_dist_indices = distances.argsort()  #排序并返回index
  class_count = {}
  for i in range(k):
    voteIlabel = training_label_set[sorted_dist_indices[i]]
    class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1 #default 0

  sorted_class_count = sorted(class_count.items(), key=lambda d:d[1], reverse=True)
  return sorted_class_count[0][0]

# test the accuracy
def knn_test(train_dataset, train_labelset, test_dataset_path, test_labelset_path, k):
  test_data_mat = sio.loadmat(test_dataset_path)
  test_label_mat = sio.loadmat(test_labelset_path)
  test_data_set = test_data_mat[list(test_data_mat.keys())[-1]]
  test_label_set = test_label_mat[list(test_label_mat.keys())[-1]]
  testset_size = test_data_set.shape[1]
  correct_classify_num = 0.0
  correct_classify_rate = 0.0
  predicted_array = []
  for i in range(testset_size):
    classify_result = knn_classify(test_data_set[:, i], train_dataset, train_labelset, k)
    predicted_array.append(int(classify_result))
    if (classify_result == test_label_set[:, i][0]):
      correct_classify_num += 1
      
  correct_classify_rate = correct_classify_num / testset_size
  print("Predicted array: \n", predicted_array)
  print("Testset Correct Classification Rate: %f%%"% (correct_classify_rate * 100))

# train the best k value through cross-validation
def training_k(training_data_set, training_label_set, validation_data_set, validation_label_set):  
  # Get the best value of k by validation
  validation_data_set_size = validation_data_set.shape[1]
  best_k = -1
  best_correct_classify_rate = 0.0
  for k in range(K_MIN, K_MAX + 1):
    print("Current value of k: ", k)
    correct_classify_num = 0
    for i in range(validation_data_set_size):
      classify_result = knn_classify(validation_data_set[:, i], training_data_set, training_label_set, k)
      if (classify_result == validation_label_set[i]):
        correct_classify_num += 1
    correct_classify_rate = float(correct_classify_num) / validation_data_set_size
    if (correct_classify_rate >= best_correct_classify_rate):
      best_k = k
      best_correct_classify_rate = correct_classify_rate
    print("Correct Classification Rate = %f%%" % (correct_classify_rate * 100))
  print("The best value of k = ", best_k)
  print("The best correct classification rate = %f%%" % (best_correct_classify_rate * 100))
  return best_k

# KNN
def knn(basepath):
  combined_set = get_combined_set(basepath + TRAINING_DATA_SUBPATH, basepath + TRAINING_LABEL_SUBPATH)
  training_data_set, training_label_set, validation_data_set, validation_label_set = get_train_validate_set(combined_set, VALIDATION_DATA_RATIO)
  best_k_value = training_k(training_data_set, training_label_set, validation_data_set, validation_label_set)
  test_training_data_set = np.c_[training_data_set, validation_data_set]
  test_training_label_set = np.hstack((training_label_set, validation_label_set))
  knn_test(test_training_data_set, test_training_label_set, basepath + TEST_DATA_SUBPATH, basepath + TEST_LABEL_SUBPATH, best_k_value)


knn(BASEPATH_BINALPHA)
