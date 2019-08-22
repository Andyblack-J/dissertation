import pandas as pd
import numpy as np
import csv
import math
import timeit
import encode
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy import arange

dataSize  = []
svm_times = []
svm_accs  = []
svm_gammas = []
svm_regs = []

def svm_run_lin():

    global reg_size

    file = open("results.txt", "a")
    file.write("\n SUPPORT VECTOR MACHINE\n")

    for c in arange(0.1, 0.6, 0.1):
        for exp in range(2, 7):
            instances = 10 ** exp
            feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
            class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
            total_data = pd.concat([feature_cols, class_cols], axis=1)

            train_size = math.floor(len(total_data)/2) # set the test-size to 50% of the entire dataset
            dataSize.append(instances)

            train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
            train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
            test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
            test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

            scale = MinMaxScaler(feature_range=(0,1)).fit(train_features)
            scaled_train = scale.transform(train_features)
            scaled_test = scale.transform(test_features)

            svm = SVC(kernel=kernel, C=c, random_state=0)
            print("\nRunning Support Vector Machine with", instances, 'instances with regularization =', c, 'using', kernel, 'kernel')
            start_svm = timeit.default_timer()
            svm.fit(scaled_train, train_class.values.ravel())
            svm_pred = svm.predict(scaled_test)
            end_svm = timeit.default_timer()
            total_time_svm = end_svm - start_svm

            svm_matrix = confusion_matrix(test_class, svm_pred)
            svm_report  = classification_report(test_class, svm_pred)

            svm_tn = svm_matrix[0][0]
            svm_fn = svm_matrix[1][0]
            svm_fp = svm_matrix[0][1]
            svm_tp = svm_matrix[1][1]

            svm_accuracy = accuracy_score(test_class, svm_pred)*100
            rounded_svm_acc = round(svm_accuracy, 3)
            rounded_svm_runtime = round(total_time_svm, 3)

            svm_runtime = [total_time_svm] # total runtime of svm
            svm_times.append(svm_runtime)
            svm_acc = [svm_accuracy] # total runtime of svm
            svm_accs.append(svm_acc)
            reg_size = [c]
            svm_regs.append(reg_size)

            svm_runtime_text = rounded_svm_runtime
            svm_accuracy_text = rounded_svm_acc

            file.write("\n\n")
            file.write(" ")
            file.write(str(instances))
            file.write(" instances\n")

            file.write("\n +-----------------------------+")

            file.write("\n | Runtime          : ")
            file.write(str(svm_runtime_text))
            file.write('  s')
            file.write("\n |..............................")
            file.write("\n | Accuracy         : ")
            file.write(str(svm_accuracy_text))
            file.write(' %')
            file.write("\n |..............................")

            file.write("\n | True Positives   : ")
            file.write(str(svm_tp))
            file.write("\n |..............................")

            file.write("\n | True Negatives   : ")
            file.write(str(svm_tn))
            file.write("\n |..............................")

            file.write("\n | False Positives  : ")
            file.write(str(svm_fp))
            file.write("\n |..............................")

            file.write("\n | False Negatives  : ")
            file.write(str(svm_fn))

            file.write("\n +-----------------------------+")

            print("--------------------------- SUPPORT VECTOR MACHINE RESULTS ---------------------------------\n")

            print("Time elapsed: [",rounded_svm_runtime, 's]\n')
            print('Kernel: ', kernel, '\n')
            print('Regularisation: ', c, '\n')
            print(svm_matrix)
            print('\n')
            print(svm_report)
            print('\n')
            print("Accuracy: ", rounded_svm_acc, '%')
            print('\n')

            print("--------------------------------------------------------------------------------------------\n")

    plt.subplot(211).set_title('SVM runtime with Linear kernel')
    plt.ylabel('time (s)')

    x1 = [0.1]
    y1 = [svm_times[3]]
    plt.scatter(x1, y1, c = 'darkkhaki', s=200)
    x2 = [0.2]
    y2 = [svm_times[7]]
    plt.scatter(x2, y2, c = 'blue', s=200)
    x3 = [0.3]
    y3 = [svm_times[11]]
    plt.scatter(x3, y3, c = 'orangered', s=200)
    x4 = [0.4]
    y4 = [svm_times[15]]
    plt.scatter(x4, y4, c = 'navy', s=200)
    x5 = [0.5]
    y5 = [svm_times[19]]
    plt.scatter(x5, y5, c = 'darkmagenta', s=200)
    plt.grid(True)

    plt.subplot(212).set_title('SVM Accuracy with Linear kernel')
    plt.xlabel('regularization (C)')
    plt.ylabel('Accuracy (%)')

    x6 = [0.1]
    y6 = [svm_accs[3]]
    plt.scatter(x6, y6, c = 'darkkhaki', s=200)
    x7 = [0.2]
    y7 = [svm_accs[7]]
    plt.scatter(x7, y7, c = 'blue', s=200)
    x8 = [0.3]
    y8 = [svm_accs[11]]
    plt.scatter(x8, y8, c = 'orangered', s=200)
    x9 = [0.4]
    y9 = [svm_accs[15]]
    plt.scatter(x9, y9, c = 'navy', s=200)
    x10 = [0.5]
    y10 = [svm_accs[19]]
    plt.scatter(x10, y10, c = 'darkmagenta', s=200)
    plt.grid(True)
    plt.legend()

    plt.show()

def svm_run_fix_lin():

    reg_size = float(input("Enter regularization value: "))

    for exp in range(2, 7):

        instances = 10 ** exp
        feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
        class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
        total_data = pd.concat([feature_cols, class_cols], axis=1)

        train_size = math.floor(len(total_data)/2) # set the test-size to 50% of the entire dataset
        dataSize.append(instances)

        train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
        train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
        test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
        test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

        scale = MinMaxScaler(feature_range=(0,1)).fit(train_features)
        scaled_train = scale.transform(train_features)
        scaled_test = scale.transform(test_features)

        svm = SVC(kernel=kernel, C=reg_size, random_state=0)
        print("\nRunning Support Vector Machine with", instances, 'instances, regularization =', reg_size, 'using', kernel, 'kernel')
        start_svm = timeit.default_timer() # start timer
        svm.fit(scaled_train, train_class.values.ravel()) # train current data on training features and class
        svm_pred = svm.predict(scaled_test) # predict target on test features
        end_svm = timeit.default_timer() # end timer
        total_time_svm = end_svm - start_svm

        svm_matrix = confusion_matrix(test_class, svm_pred)
        svm_report  = classification_report(test_class, svm_pred)

        svm_tn = svm_matrix[0][0]
        svm_fn = svm_matrix[1][0]
        svm_fp = svm_matrix[0][1]
        svm_tp = svm_matrix[1][1]

        svm_accuracy = accuracy_score(test_class, svm_pred)*100
        rounded_svm_acc = round(svm_accuracy, 3)
        rounded_svm_runtime = round(total_time_svm, 3)

        svm_runtime = [total_time_svm] # total runtime of svm
        svm_times.append(svm_runtime)
        svm_acc = [svm_accuracy] # total runtime of svm
        svm_accs.append(svm_acc)
        reg_sizes = [reg_size]
        svm_regs.append(reg_size)

        svm_runtime_text = rounded_svm_runtime
        svm_accuracy_text = rounded_svm_acc

        print("--------------------------- SUPPORT VECTOR MACHINE RESULTS ---------------------------------\n")

        print("Time elapsed: [",rounded_svm_runtime, 's]\n')
        print('Kernel: ', kernel, '\n')
        print('Regularisation: ', reg_size, '\n')
        print(svm_matrix)
        print('\n')
        print(svm_report)
        print('\n')
        print("Accuracy: ", rounded_svm_acc, '%')
        print('\n')

        print("--------------------------------------------------------------------------------------------\n")

    plt.show()
    plt.subplot(211).set_title('SVM runtime with Linear kernel')
    plt.ylabel('time (s)')

    x1 = [0.1]
    y1 = [svm_times[0]]
    plt.scatter(x1, y1, c = 'darkkhaki', s=200)
    x2 = [0.2]
    y2 = [svm_times[1]]
    plt.scatter(x2, y2, c = 'blue', s=200)
    x3 = [0.3]
    y3 = [svm_times[2]]
    plt.scatter(x3, y3, c = 'orangered', s=200)
    x4 = [0.4]
    y4 = [svm_times[3]]
    plt.scatter(x4, y4, c = 'navy', s=200)
    x5 = [0.5]
    y5 = [svm_times[4]]
    plt.scatter(x5, y5, c = 'darkmagenta', s=200)
    plt.grid(True)

    plt.subplot(212).set_title('SVM Accuracy with 0.3 regularization')
    plt.xlabel('data size')
    plt.ylabel('Accuracy (%)')
    plt.xscale('log') # X axis = 10^1 - 10^6

    x6 = [0.1]
    y6 = [svm_accs[0]]
    plt.scatter(x6, y6, c = 'darkkhaki', s=200)
    x7 = [0.2]
    y7 = [svm_accs[1]]
    plt.scatter(x7, y7, c = 'darkkhaki', s=200)
    x8 = [0.3]
    y8 = [svm_accs[2]]
    plt.scatter(x8, y8, c = 'orangered', s=200)
    x9 = [0.4]
    y9 = [svm_accs[3]]
    plt.scatter(x9, y9, c = 'navy', s=200)
    x10 = [0.5]
    y10 = [svm_accs[4]]
    plt.scatter(x10, y10, c = 'darkmagenta', s=200)

    plt.grid(True)
    plt.legend()

    plt.show()

def svm_run_rbf():

    file = open("results.txt", "a")
    file.write("\n SUPPORT VECTOR MACHINE\n")

    for g in arange(0.1, 0.6, 0.1):
        for exp in range(2, 7):
            instances = 10 ** exp
            feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
            class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
            total_data = pd.concat([feature_cols, class_cols], axis=1)

            train_size = math.floor(len(total_data)/2) # set the test-size to 50% of the entire dataset
            dataSize.append(instances)

            train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
            train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
            test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
            test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

            scale = MinMaxScaler(feature_range=(0,1)).fit(train_features)
            scaled_train = scale.transform(train_features)
            scaled_test = scale.transform(test_features)

            svm = SVC(kernel=kernel, gamma=g, random_state=0)
            print("\nRunning Support Vector Machine with", instances, 'instances with gamma =', g, 'using', kernel, 'kernel')
            start_svm = timeit.default_timer()
            svm.fit(scaled_train, train_class.values.ravel())
            svm_pred = svm.predict(scaled_test)
            end_svm = timeit.default_timer()
            total_time_svm = end_svm - start_svm

            svm_matrix = confusion_matrix(test_class, svm_pred)
            svm_report  = classification_report(test_class, svm_pred)

            svm_tn = svm_matrix[0][0]
            svm_fn = svm_matrix[1][0]
            svm_fp = svm_matrix[0][1]
            svm_tp = svm_matrix[1][1]

            #svm_accuracy = ((svm_tp + svm_tn) / (svm_tp + svm_tn + svm_fp + svm_fn))*100
            svm_accuracy = accuracy_score(test_class, svm_pred)*100
            rounded_svm_acc = round(svm_accuracy, 3)
            rounded_svm_runtime = round(total_time_svm, 3)

            svm_runtime = [total_time_svm] # total runtime of svm
            svm_times.append(svm_runtime)
            svm_acc = [svm_accuracy] # total runtime of svm
            svm_accs.append(svm_acc)
            gamma_sizes = [g]
            svm_gammas.append(g)

            svm_runtime_text = rounded_svm_runtime
            svm_accuracy_text = rounded_svm_acc

            file.write("\n\n")
            file.write(" ")
            file.write(str(instances))
            file.write(" instances\n")

            file.write("\n +-----------------------------+")

            file.write("\n | Runtime          : ")
            file.write(str(svm_runtime_text))
            file.write('  s')
            file.write("\n |..............................")
            file.write("\n | Accuracy         : ")
            file.write(str(svm_accuracy_text))
            file.write(' %')
            file.write("\n |..............................")

            file.write("\n | True Positives   : ")
            file.write(str(svm_tp))
            file.write("\n |..............................")

            file.write("\n | True Negatives   : ")
            file.write(str(svm_tn))
            file.write("\n |..............................")

            file.write("\n | False Positives  : ")
            file.write(str(svm_fp))
            file.write("\n |..............................")

            file.write("\n | False Negatives  : ")
            file.write(str(svm_fn))

            file.write("\n +-----------------------------+")

            print("--------------------------- SUPPORT VECTOR MACHINE RESULTS ---------------------------------\n")

            print("Time elapsed: [",rounded_svm_runtime, 's]\n')
            print('Kernel: ', kernel, '\n')
            print('Gamma: ', g, '\n')
            #print(svm_matrix)
            print('\n')
            print(svm_report)
            print('\n')
            print("Accuracy: ", rounded_svm_acc, '%')
            print('\n')

            print("--------------------------------------------------------------------------------------------\n")

    plt.subplot(211).set_title('SVM runtime with RBF kernel')
    plt.ylabel('time (s)')

    x1 = [0.1]
    y1 = [svm_times[3]]
    plt.scatter(x1, y1, c = 'darkkhaki', s=200)
    x2 = [0.2]
    y2 = [svm_times[7]]
    plt.scatter(x2, y2, c = 'darkkhaki', s=200)
    x3 = [0.3]
    y3 = [svm_times[11]]
    plt.scatter(x3, y3, c = 'orangered', s=200)
    x4 = [0.4]
    y4 = [svm_times[15]]
    plt.scatter(x4, y4, c = 'navy', s=200)
    x5 = [0.5]
    y5 = [svm_times[19]]
    plt.scatter(x5, y5, c = 'darkmagenta', s=200)
    plt.grid(True)

    plt.subplot(212).set_title('SVM Accuracy with RBF kernel')
    plt.xlabel('gamma (g)')
    plt.ylabel('Accuracy (%)')

    x6 = [0.1]
    y6 = [svm_accs[3]]
    plt.scatter(x6, y6, c = 'darkkhaki', s=200)
    x7 = [0.2]
    y7 = [svm_accs[7]]
    plt.scatter(x7, y7, c = 'darkkhaki', s=200)
    x8 = [0.3]
    y8 = [svm_accs[11]]
    plt.scatter(x8, y8, c = 'orangered', s=200)
    x9 = [0.4]
    y9 = [svm_accs[15]]
    plt.scatter(x9, y9, c = 'navy', s=200)
    x10 = [0.5]
    y10 = [svm_accs[19]]
    plt.scatter(x10, y10, c = 'darkmagenta', s=200)

    plt.grid(True)
    plt.legend()

    plt.show()
    file.close()

def svm_run_fix_rbf():

    global gamma_size
    gamma_size = float(input("Enter gamma value: "))
    for exp in range(2, 6):
        instances = 10 ** exp
        feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
        class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
        total_data = pd.concat([feature_cols, class_cols], axis=1)

        train_size = math.floor(len(total_data)/4) # set the test-size to 50% of the entire dataset
        dataSize.append(instances)

        train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
        train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
        test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
        test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

        scale = MinMaxScaler(feature_range=(0,1)).fit(train_features)
        scaled_train = scale.transform(train_features)
        scaled_test = scale.transform(test_features)

        svm = SVC(kernel=kernel, gamma=gamma_size, random_state=0)
        print("\nRunning Support Vector Machine with", instances, 'instances with gamma =', gamma_size, 'using', kernel, 'kernel')
        start_svm = timeit.default_timer()
        svm.fit(scaled_train, train_class.values.ravel())
        svm_pred = svm.predict(scaled_test)
        end_svm = timeit.default_timer()
        total_time_svm = end_svm - start_svm

        svm_matrix = confusion_matrix(test_class, svm_pred)
        svm_report  = classification_report(test_class, svm_pred)

        svm_tn = svm_matrix[0][0]
        svm_fn = svm_matrix[1][0]
        svm_fp = svm_matrix[0][1]
        svm_tp = svm_matrix[1][1]

        svm_accuracy = ((svm_tp + svm_tn) / (svm_tp + svm_tn + svm_fp + svm_fn))*100
        rounded_svm_acc = round(svm_accuracy, 3)
        rounded_svm_runtime = round(total_time_svm, 3)

        svm_runtime = [total_time_svm] # total runtime of svm
        svm_times.append(svm_runtime)
        svm_acc = [svm_accuracy] # total runtime of svm
        svm_accs.append(svm_acc)
        gamma_size = [gamma_size]
        svm_gammas.append(gamma_size)

        svm_runtime_text = rounded_svm_runtime
        svm_accuracy_text = rounded_svm_acc

    plt.subplot(211).set_title('SVM runtime with gamma tuning')
    plt.ylabel('time (s)')
    plt.xscale('log') # X axis = 10^1 - 10^6
    x1 = [100]
    y1 = [svm_times[0]]
    plt.scatter(x1, y1, c = 'darkkhaki', s=500, label='i = 100')
    x2 = [1000]
    y2 = [svm_times[1]]
    plt.scatter(x2, y2, c = 'darkkhaki', s=500, label='i = 1000')
    x3 = [10000]
    y3 = [svm_times[2]]
    plt.scatter(x3, y3, c = 'orangered', s=500, label='i = 10000')
    x4 = [100000]
    y4 = [svm_times[3]]
    plt.scatter(x4, y4, c = 'navy', s=500, label='i = 100000')
    x5 = [1000000]
    y5 = [svm_times[4]]
    plt.scatter(x5, y5, c = 'darkmagenta', s=500, label='i = 1000000')
    plt.grid(True)

    plt.subplot(212).set_title('SVM Accuracy with regularization tuning')
    plt.xlabel('data size')
    plt.ylabel('Accuracy')
    plt.xscale('log') # X axis = 10^1 - 10^6
    x6 = [100]
    y6 = [svm_accs[0]]
    plt.scatter(x6, y6, c = 'darkkhaki', s=5000, label='i = 100')
    x7 = [1000]
    y7 = [svm_accs[1]]
    plt.scatter(x7, y7, c = 'darkkhaki', s=5000, label='i = 1000')
    x8 = [10000]
    y8 = [svm_accs[2]]
    plt.scatter(x8, y8, c = 'orangered', s=500, label='i = 10000')
    x9 = [100000]
    y9 = [svm_accs[3]]
    plt.scatter(x9, y9, c = 'navy', s=5000, label='i = 100000')
    x10 = [1000000]
    y10 = [svm_accs[4]]
    plt.scatter(x10, y10, c = 'darkmagenta', s=500, label='i = 1000000')
    plt.grid(True)
    plt.legend()

    plt.show()
    file.close()

print("[kernel option 1: Linear kernel with auto adjusted C]\n")
print("[kernel option 2: RBF kernel with auto adjusted gamma]\n")
print("[kernel option 3: Linear kernel with fixed C]\n")
print("[kernel option 4: RBF kernel with fixed gamma]\n")

kernel = input("Kernel: ") # Kernel type used in SVM classification
option = input("Fixed or auto tune: ")

if kernel == 'linear' and option == 'auto':
    print("Running classifier with regularization tuning 0.1 - 0.5")
    svm_run_lin()

if kernel == 'rbf' and option == 'auto':
    print("Running classifier with gamma tuning from 0.1 - 0.5")
    svm_run_rbf()

if kernel == 'linear' and option == 'fixed':
    print("Running classifier with fixed regularization tuning")
    svm_run_fix_lin()

if kernel == 'rbf' and option == 'fixed':
    print("Running classifier with fixed gamma tuning")
    svm_run_fix_rbf()
