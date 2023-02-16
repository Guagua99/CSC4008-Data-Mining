import numpy as np
import scipy.io as sio
import pandas as pd

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
    return training_set, training_data_set, training_label_set, validation_set, validation_data_set, validation_label_set

# density classifier
def density_classify(input_data, training_data_set, training_label_set, r):
    training_data_set_size = training_data_set.shape[1]
    input_data = input_data.reshape(input_data.shape[0], -1) #644*1
    diff_mat = np.tile(input_data, (1, training_data_set_size)) - training_data_set #644*272
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=0)
    distances = sq_distances ** 0.5  #欧式距离

    print(min(distances))
    
    class_count = {}
    for j in range(distances.shape[0]):
        if (distances[j]<=r):
            voteIlabel = training_label_set[j]
            class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
        else:
            continue

    sorted_class_count = sorted(class_count.items(), key=lambda d:d[1], reverse=True)
    if (sorted_class_count == []):
        classify_result = 0
    else:
        classify_result = sorted_class_count[0][0]
    return classify_result

# test the accuracy
def density_test(train_dataset, train_labelset, test_dataset_path, test_labelset_path,r):
    test_data_mat = sio.loadmat(test_dataset_path)
    test_label_mat = sio.loadmat(test_labelset_path)
    test_data_set = test_data_mat[list(test_data_mat.keys())[-1]]
    test_label_set = test_label_mat[list(test_label_mat.keys())[-1]]
    testset_size = test_data_set.shape[1]
    correct_classify_num = 0.0
    correct_classify_rate = 0.0
    predicted_array = []
    for i in range(testset_size):
        classify_result = density_classify(test_data_set[:, i], train_dataset, train_labelset,r)
        predicted_array.append(int(classify_result))
        if (classify_result == test_label_set[:, i][0]):
            correct_classify_num += 1

    correct_classify_rate = correct_classify_num / testset_size
    print("Predicted array: \n", predicted_array)
    print("Testset Correct Classification Rate: %f%%"% (correct_classify_rate * 100))

def density(basepath):
    combined_set = get_combined_set(basepath + TRAINING_DATA_SUBPATH, basepath + TRAINING_LABEL_SUBPATH)
    training_set, training_data_set, training_label_set, validation_set, validation_data_set, validation_label_set = get_train_validate_set(combined_set, VALIDATION_DATA_RATIO)
    density_test(training_data_set, training_label_set, basepath + TEST_DATA_SUBPATH, basepath + TEST_LABEL_SUBPATH,500)


density(BASEPATH_ATNT)

