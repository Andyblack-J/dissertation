import pandas as pd
import numpy as np
import csv
import math
import timeit
import random
import encode
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataSize  = []
knn_times = []
knn_accs = []
prec_avg = []
k_size   = []

def knn_run_auto():

    global knn_times
    global knn_accs

    file = open("results.txt", "a")
    file.write("\n K NEAREST NEIGHBOURS\n")

    for neighbours in range (10, 60, 10):
        for exp in range(2, 7):

            instances = 10 ** exp
            feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
            class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
            total_data = pd.concat([feature_cols, class_cols], axis=1)

            print("Classifying data with", instances, "instances with k =", neighbours, "and train:test split = 67:33")

            train_size = math.floor(len(total_data)/2) # set the test-size to 50% of the entire dataset
            dataSize.append(instances)

            train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
            train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
            test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
            test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

            print("\nRunning K-Nearest Neighbours...")
            start_knn = timeit.default_timer() # start timer
            knn = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform')
            knn.fit(train_features, train_class.values.ravel()) # train current data on training features and class
            knn_pred = knn.predict(test_features) # predict target on test features
            end_knn = timeit.default_timer() # end timer
            total_time_knn = end_knn - start_knn
            knn_matrix = confusion_matrix(test_class, knn_pred)
            knn_report = classification_report(test_class, knn_pred)

            knn_tn = knn_matrix[0][0] # true negatives
            knn_fn = knn_matrix[1][0] # false negatives
            knn_fp = knn_matrix[0][1] # false positives
            knn_tp = knn_matrix[1][1] # true positives

            knn_accuracy = accuracy_score(test_class, knn_pred)*100
            rounded_knn_acc = round(knn_accuracy, 3)
            rounded_knn_runtime = round(total_time_knn, 3)

            knn_runtime = [rounded_knn_runtime] # total runtime of knn
            knn_times.append(knn_runtime)

            knn_acc = [rounded_knn_acc]
            knn_accs.append(knn_acc)

            knn_runtime_text = rounded_knn_runtime
            knn_accuracy_text = rounded_knn_acc

            file.write("\n\n")
            file.write(" ")
            file.write(str(instances))
            file.write(" instances\n")

            file.write(" ")
            file.write(str(neighbours))
            file.write(" neighbours\n")

            file.write("\n +-----------------------------+")

            file.write("\n | Runtime          : ")
            file.write(str(knn_runtime_text))
            file.write('  s')
            file.write("\n |..............................")
            file.write("\n | Accuracy         : ")
            file.write(str(knn_accuracy_text))
            file.write(' %')
            file.write("\n |..............................")

            file.write("\n | True Positives   : ")
            file.write(str(knn_tp))
            file.write("\n |..............................")

            file.write("\n | True Negatives   : ")
            file.write(str(knn_tn))
            file.write("\n |..............................")

            file.write("\n | False Positives  : ")
            file.write(str(knn_fp))
            file.write("\n |..............................")

            file.write("\n | False Negatives  : ")
            file.write(str(knn_fn))

            file.write("\n +-----------------------------+")

            print("------------------------------ K-NEAREST NEIGHBOURS RESULTS --------------------------------\n")

            #print("Time elapsed: [{} m {} secs]".format(knn_minutes, knn_seconds))
            print("Time elapsed: [", rounded_knn_runtime, "]")
            print('Neighbours (k): ', neighbours, '\n')
            print(knn_matrix)
            print('\n')
            print(knn_report)
            print('\n')
            print("Accuracy: ", rounded_knn_acc, '%')
            print('\n')

            print("--------------------------------------------------------------------------------------------\n")

    plt.rc('font', family = 'serif', size=13)
    plt.subplot(211).set_title('KNN runtime')
    plt.ylabel('time (s)')

    x1 = [10]
    y1 = [knn_times[3]]
    plt.scatter(x1, y1, s=200, c = 'darkkhaki')

    x2 = [20]
    y2 = [knn_times[7]]
    plt.scatter(x2, y2, s=200, c = 'orangered')

    x3 = [30]
    y3 = [knn_times[11]]
    plt.scatter(x3, y3, s=200, c = 'navy')

    x4 = [40]
    y4 = [knn_times[15]]
    plt.scatter(x4, y4, s=200, c = 'darkmagenta')

    x5 = [50]
    y5 = [knn_times[19]]
    plt.scatter(x5, y5, s=200, c = 'green')

    plt.grid(True)

    plt.rc('font', family = 'serif', size=13)
    plt.subplot(212).set_title('KNN Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Neighbours`')

    x6 = [10]
    y6 = [knn_accs[3]]
    plt.scatter(x26, y26, s=200, c = 'darkkhaki')

    x7 = [20]
    y7 = [knn_accs[7]]
    plt.scatter(x27, y27, s=200, c = 'orangered')

    x8 = [30]
    y8 = [knn_accs[11]]
    plt.scatter(x28, y28, s=200, c = 'navy')

    x9 = [40]
    y9 = [knn_accs[15]]
    plt.scatter(x29, y29, s=200, c = 'darkmagenta')

    x10 = [50]
    y10 = [knn_accs[19]]
    plt.scatter(x30, y30, s=200, c = 'green')

    plt.grid(True)

    plt.legend()
    plt.show()

def knn_run_fixed():

    global knn_times
    global knn_accs

    file = open("results.txt", "a")
    file.write("\n K NEAREST NEIGHBOURS\n")
    neighbours = int(input("Enter number of neighbours: "))

    for exp in range(2, 7):
        instances = 10 ** exp
        feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
        class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
        total_data = pd.concat([feature_cols, class_cols], axis=1)

        print(total_data)

        print('Classifying data on', instances, 'instances with k =', neighbours)

        train_size = math.floor(len(total_data)/2) # set the test-size to 50% of the entire dataset
        dataSize.append(instances)
        train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
        train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
        test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
        test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

        print("\nRunning K-Nearest Neighbours...")
        start_knn = timeit.default_timer() # start timer
        knn = KNeighborsClassifier(n_neighbors=neighbours, weights='uniform')
        knn.fit(train_features, train_class.values.ravel()) # train current data on training features and class
        knn_pred = knn.predict(test_features) # predict target on test features
        end_knn = timeit.default_timer() # end timer
        total_time_knn = end_knn - start_knn

        knn_matrix = confusion_matrix(test_class, knn_pred)
        knn_report = classification_report(test_class, knn_pred)

        knn_tn = knn_matrix[0][0] # true negatives
        knn_fn = knn_matrix[1][0] # false negatives
        knn_fp = knn_matrix[0][1] # false positives
        knn_tp = knn_matrix[1][1] # true positives

        knn_accuracy = accuracy_score(test_class, knn_pred)*100
        rounded_knn_acc = round(knn_accuracy, 3)
        rounded_knn_runtime = round(total_time_knn, 3)

        knn_runtime = [total_time_knn] # total runtime of knn
        knn_times.append(knn_runtime)

        knn_acc = [rounded_knn_acc]
        knn_accs.append(knn_acc)

        knn_runtime_text = rounded_knn_runtime
        knn_accuracy_text = rounded_knn_acc

        print("------------------------------ K-NEAREST NEIGHBOURS RESULTS --------------------------------\n")

        print("Time elapsed: [",rounded_knn_runtime, 's]\n')
        print('Neighbours (k): ', neighbours, '\n')
        print(knn_matrix)
        print('\n')
        print(knn_report)
        print('\n')
        print("Accuracy: ", rounded_knn_acc, '%')
        print('\n')

        print("--------------------------------------------------------------------------------------------\n")

    plt.rc('font', family = 'serif', size=13)
    plt.subplot(211).set_title("KNN runtime with 30 neighbours")
    plt.ylabel('time (s)')
    plt.xscale('log') # X axis = 10^1 - 10^6

    x1 = [100]
    y1 = [knn_times[0]]
    plt.scatter(x1, y1, s=200, c = 'darkkhaki')

    x2 = [1000]
    y2 = [knn_times[1]]
    plt.scatter(x2, y2, s=200, c = 'orangered')

    x3 = [10000]
    y3 = [knn_times[2]]
    plt.scatter(x3, y3, s=200, c = 'navy')

    x4 = [100000]
    y4 = [knn_times[3]]
    plt.scatter(x4, y4, s=200, c = 'darkmagenta')

    x5 = [1000000]
    y5 = [knn_times[4]]
    plt.scatter(x5, y5, s=200, c = 'red')
    plt.grid(True)

    plt.rc('font', family = 'serif', size=13)
    plt.subplot(212).set_title('KNN Accuracy with 30 neighbours')
    plt.ylabel('Accuracy (%)')
    plt.xscale('log') # X axis = 10^1 - 10^6
    plt.xlabel('data size')

    x6 = [100]
    y6 = [knn_accs[0]]
    plt.scatter(x6, y6, s=200, c = 'darkkhaki')

    x7 = [1000]
    y7 = [knn_accs[1]]
    plt.scatter(x7, y7, s=200, c = 'orangered')

    x8 = [10000]
    y8 = [knn_accs[2]]
    plt.scatter(x8, y8, s=200, c = 'navy')

    x9 = [100000]
    y9 = [knn_accs[3]]
    plt.scatter(x9, y9, s=200, c = 'darkmagenta')

    x10 = [1000000]
    y10 = [knn_accs[4]]
    plt.scatter(x10, y10, s=200, c = 'green')
    plt.grid(True)

    plt.legend()
    plt.show()


print("[option 1: Incremented k]\n")
print("[option 2: Fixed k]\n")
print("For option 1, enter 'knn'\n")
print("For option 2, enter 'fixed k'\n")

option = input("Select option: ")

if option == 'knn':
    knn_run_auto()

elif option == 'fixed k':
    knn_run_fixed()
