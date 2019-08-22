import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import timeit

from multiprocessing import Process

knn_times    = []
nb_times     = []
svm_times    = []

knn_accs     = []
nb_accs      = []
svm_accs     = []

dataSize     = []

feature_cols = []
class_cols =   []

global neighbours
global kernel
global instances

def encodeData():

    names = ['protocol', 'range(m)', 'power_src', 'weight(g)', 'processing_power(ghz)', 'device_type'] # column headings
    file = pd.read_csv('dataset.csv', names=names)

    non_enc = pd.DataFrame(file, columns=['range(m)', 'weight(g)', 'processing_power(ghz)', 'device_type']) # columns that do not require encoding
    enc_cols = pd.DataFrame(file, columns=['protocol', 'power_src']) # columns that require Dummy encoding
    new_cols = pd.get_dummies(enc_cols, columns=enc_cols) # encode 'protocol' and 'power_src' columns
    for col in new_cols:
        new_cols[col] = new_cols[col].astype(object) # change the type of newly encoded columns to object
    concat_datasets = pd.concat([non_enc, new_cols], axis=1) # concatenate the encoded columns with non-encoded columns

    cols = list(concat_datasets.columns.values)
    cols.pop(cols.index('device_type')) # pop the column 'type' from the list of columns
    new_dataset = concat_datasets[cols+['device_type']] # append 'type' to end of list
    feature_cols = new_dataset.iloc[:, :-1] # all features excluding final column
    class_cols = new_dataset.iloc[:, -1:] # final column excluding feature columns

    neighbours = int(input("Select number of neighbours: ")) # number of neighbours to be included in KNN prediction
    kernel = str(input("Kernel: ")) # Kernel type used in SVM classification

    return feature_cols, class_cols

def splitData(feature_cols, class_cols):
    total_data = feature_cols + class_cols
    for exp in range(3, 7):
        instances = 10 ** exp
        print('Classifying data with ', instances, 'instances')
        fileSize = total_data.iloc[:instances]
        train_size = math.floor(len(fileSize)/2) # set the test-size to 50% of the entire dataset
        test_size = math.floor(fileSize - train_size) # set the test-size to 50% of the entire dataset
        dataSize.append(instances)

        train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
        train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
        test_features = feature_cols.iloc[test_size:] # set test features from midpoint onwards
        test_class = class_cols.iloc[test_size:] # set test class from midpoint onwards

    return dataSize, train_features, train_class, test_features, test_class, total_data
