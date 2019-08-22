import pandas as pd
import matplotlib.pyplot as plt
import nb, knn, svm

plt.rc('font', family = 'serif', size=13)
plt.subplot(211).set_title('Classifier runtime')
plt.rc('font', family = 'serif', size=13)
plt.legend(['KNN', 'NB', 'SVM'], loc='lower center')
plt.title("Classifier run times")
plt.ylabel('time (s)')
plt.xscale('log') # X axis = 10^1 - 10^6

x2 = [100, 1000, 10000, 100000, 1000000]
y2 = [nb.nb_times[0], nb.nb_times[1], nb.nb_times[2], nb.nb_times[3], nb.nb_times[4]]
plt.plot(x2, y2, c = 'orangered', marker='o', linestyle=':', linewidth=3.0, label='NB runtime')

x1 = [100, 1000, 10000, 100000, 1000000]
y1 = [knn.knn_times[0], knn.knn_times[1], knn.knn_times[2], knn.knn_times[3],  knn.knn_times[4]]
plt.plot(x1, y1, c='darkkhaki', marker='o', linestyle='-', linewidth=3.0, label='KNN runtime')

x3 = [100, 1000, 10000, 100000, 1000000]
y3 = [svm.svm_times[0], svm.svm_times[1], svm.svm_times[2], svm.svm_times[3], svm.svm_times[4]]
plt.plot(x3, y3, c = 'green', marker='o', linestyle='--', label='SVM runtime')

plt.grid(True)
plt.legend()

plt.rc('font', family = 'serif', size=13)
plt.subplot(212).set_title('Classifier accuracy')
plt.rc('font', family = 'serif', size=13)
plt.legend(['KNN', 'NB', 'SVM'], loc='lower center')
plt.title("Classifier accuracy")
plt.ylabel('Accuracy (%)')
plt.xscale('log') # X axis = 10^1 - 10^6

x1 = [100, 1000, 10000, 100000, 1000000]
y1 = [knn.knn_accs[0], knn.knn_accs[1], knn.knn_accs[2], knn.knn_accs[3], knn.knn_accs[4]] #add 19 back
plt.plot(x1, y1, c='darkkhaki', marker='o', linestyle='-', linewidth=3.0, label='KNN accuracy')

x2 = [100, 1000, 10000, 100000, 1000000]
y2 = [nb.nb_accs[0], nb.nb_accs[1], nb.nb_accs[2], nb.nb_accs[3], nb.nb_accs[4]]
plt.plot(x2, y2, c = 'orangered', marker='o', linestyle=':', linewidth=3.0, label='NB accuracy')

x3 = [100, 1000, 10000, 100000, 1000000]
y3 = [svm.svm_accs[0], svm.svm_accs[1], svm.svm_accs[2], svm.svm_accs[3], svm.svm_accs[4]]
plt.plot(x3, y3, c = 'navy', marker='o', linestyle='--', label='SVM accuracy')

plt.grid(True)
plt.legend()

plt.show()
