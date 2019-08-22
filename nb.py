import pandas as pd
import numpy as np
import csv
import math
import timeit
import encode
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataSize  = []
nb_times = []
nb_accs  = []

def nb_run():

    file = open("results.txt", "a")
    file.write("\nNAIVE BAYES\n")

    for exp in range(2, 7):
        instances = 10 ** exp
        feature_cols = encode.new_dataset.iloc[:instances, :-1] # all features excluding final column
        class_cols = encode.new_dataset.iloc[:instances, -1:] # final column excluding feature columns
        total_data = pd.concat([feature_cols, class_cols], axis=1)

        print('Classifying data with', instances, 'instances')

        train_size = math.floor(len(total_data)/3) # set the test-size to 50% of the entire dataset
        dataSize.append(instances)

        train_features = feature_cols.iloc[:train_size] # set training features up to 50% of dataset
        train_class = class_cols.iloc[:train_size] # set training class up to 50% of dataset
        test_features = feature_cols.iloc[train_size:] # set test features from midpoint onwards
        test_class = class_cols.iloc[train_size:] # set test class from midpoint onwards

        nb = GaussianNB()
        print("\nRunning Naive Bayes...")
        start_nb = timeit.default_timer() # start timer
        nb.fit(train_features, train_class.values.ravel()) # train current data on training features and class
        nb_pred = nb.predict(test_features) # predict target on test features
        end_nb = timeit.default_timer() # end timer
        total_time_nb = (end_nb - start_nb)
        print("\nClassification complete.")

        nb_matrix = confusion_matrix(test_class, nb_pred)
        nb_report = classification_report(test_class, nb_pred)

        nb_tn =  nb_matrix[0][0]
        nb_fn =  nb_matrix[1][0]
        nb_fp =  nb_matrix[0][1]
        nb_tp =  nb_matrix[1][1]

        #nb_accuracy = ((nb_tp + nb_tn) / (nb_tp + nb_tn + nb_fp + nb_fn))*100
        nb_accuracy = accuracy_score(test_class, nb_pred)*100
        rounded_nb_acc = round(nb_accuracy, 3)
        rounded_nb_runtime = round(total_time_nb, 3)

        nb_runtime = [total_time_nb]   # total runtime of nb
        nb_times.append(nb_runtime)

        nb_acc = [rounded_nb_acc]   # total runtime of nb
        nb_accs.append(nb_acc)

        nb_runtime_text = rounded_nb_runtime
        nb_accuracy_text = rounded_nb_acc

        file.write("\n\n")
        file.write(" ")
        file.write(str(instances))
        file.write(" instances\n")

        file.write("\n +-----------------------------+")

        file.write("\n | Runtime          : ")
        file.write(str(nb_runtime_text))
        file.write('  s')
        file.write("\n |..............................")

        file.write("\n | Accuracy         : ")
        file.write(str(nb_accuracy_text))
        file.write(' %')
        file.write("\n |..............................")

        file.write("\n | True Positives   : ")
        file.write(str(nb_tp))
        file.write("\n |..............................")

        file.write("\n | True Negatives   : ")
        file.write(str(nb_tn))
        file.write("\n |..............................")

        file.write("\n | False Positives  : ")
        file.write(str(nb_fp))
        file.write("\n |..............................")

        file.write("\n | False Negatives  : ")
        file.write(str(nb_fn))

        file.write("\n +-----------------------------+")


        print("-------------------------------- NAIVE BAYES RESULTS ---------------------------------------\n")

        print("[Time elapsed: [", rounded_nb_runtime, 's]\n')
        print(nb_matrix)
        print('\n')
        print(nb_report)
        print('\n')
        print("Accuracy: ", rounded_nb_acc, '%')
        print('\n')

        print("--------------------------------------------------------------------------------------------\n")

    plt.subplot(211).set_title('NB runtime')
    plt.ylabel('time (s)')
    plt.xscale('log') # X axis = 10^1 - 10^6
    plt.plot(dataSize, nb_times, c = 'red', label='NB runtime')
    plt.grid(True)
    plt.legend()

    plt.subplot(212).set_title('NB Accuracy')
    plt.xlabel('data size')
    plt.ylabel('Accuracy (%)')
    plt.xscale('log') # X axis = 10^1 - 10^6
    plt.plot(dataSize, nb_accs, c = 'blue', label='NB accuracy')
    plt.grid(True)
    plt.legend()

    plt.show()
    file.close()

nb_run()
