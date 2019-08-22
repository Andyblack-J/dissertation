A COMPARATIVE STUDY OF CLASSIFICATION ALGORITHMS TO MANAGE IOT DEVICES
--------------------------------------------------------------------------------
The files knn.py, nb.py, svm.py, encoder.py and dataset.py demonstrate the
application of machine learning techniques to a large dataset. dataset.py
generates a synthetic dataset in which up-to one million (10^6) samples are
created at random. The output from this file is provided as input to encoder.py
such that some columns are encoded in order to be fit for purpose when applied
to machine learning problems. The output from the encoder is then given as input
to each classifier (knn.py, nb.py and svm.py) individually to test the
performance of each algorithm and determine the most optimal classification
technique for the given problem.
--------------------------------------------------------------------------------
================================================================================
                                GETTING STARTED
================================================================================
Prerequisites - Python3, pip

To check if pip is installed:

  $ command -v pip

How to install pip on Linux/macOS:

  $ pip install -U pip
--------------------------------------------------------------------------------
How to install Python3 on Ubuntu 16.10 or later (pip required):

  $ sudo apt-get update
  $ sudo apt-get install python3.6

To launch Python3 interpreter on Linux/macOS:

  $ python3
--------------------------------------------------------------------------------

================================================================================
                              RUNNING THE TESTS
================================================================================
Order of execution:
  dataset.py -> encoder.py -> knn.py
                           -> nb.py
                           -> svm.py
-------------------------------------------------------------------------------
To generate dataset:
  $ python3 dataset.py

To encode the dataset:
  $ python3 encoder.py

To execute K-Nearest Neighbours classifier:
  $ python3 knn.py

To execute Naive Bayes classifier:
  $ python3 nb.py

To execute Support Vector Machine classifier:
  $ python3 svm.py
--------------------------------------------------------------------------------
================================================================================
                                  AUTHOR
================================================================================

Corey Joshua Fielding

================================================================================
                              ACKNOWLEDGEMENTS
================================================================================

Scott Robinson(2018)K-Nearest Neighbors Algorithm in Python and Scikit-Learn[source code]. https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

Vik Paruchuri(2015)Tutorial: K Nearest Neighbors in Python[source code].
https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

Usman Malik(2018)Implementing SVM and Kernel SVM with Python's Scikit-Learn[source code].
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

Krunal(2018)How To Prepare Your Dataset For Machine Learning In Python[source code].
https://appdividend.com/2018/07/23/prepare-dataset-for-machine-learning-in-python/

Rowan Langford(2017)The Dummy’s Guide to Creating Dummy Variables[source code].
https://towardsdatascience.com/the-dummys-guide-to-creating-dummy-variables-f21faddb1d40

Charlie Haley(2016)move column in pandas dataframe[source code].
https://stackoverflow.com/a/35321983
