# -*- coding: utf-8 -*-

"""
@author: Hugo Moritz
"""

# Import the classification algorithms for the machine learning models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Import functions
from model import train

def classification(vectorized_data):
    """Trains and tests the specified machine learning classifiers with the 
    vectorized data.

    Parameters
    ----------
    vectorized_data: numpy.Arrays
        x_train, x_test, y_train, y_test rf-idf vectorized variables

    """

    '''
        Parameters for Grid Search CV optimisation. Uncomment and remove/uncomment
        the other parameter.
    '''
    NB_params = None
    #SVM_params = {'kernel':['linear'], 'C': list(range(1,20)), 'gamma': [1,0.1,0.01,0.001]}
    SVM_params = None
    #KNN_params = {'n_neighbors': list(range(1,31)), 'weights': ["uniform", "distance"]}
    KNN_params = None
    #NN_params = {'hidden_layer_sizes': [(2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}
    NN_params = None
    #DT_params = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
    DT_params = None
    
    ''' Executing train function with model (estimator), vectorized data and 
        parameters
    '''
    train((MultinomialNB()), vectorized_data, NB_params)
    train((SVC(kernel='linear', C=4)), vectorized_data, SVM_params)
    train((KNeighborsClassifier(n_neighbors=4, weights='distance')), vectorized_data, KNN_params)
    train((MLPClassifier(hidden_layer_sizes=(8,), alpha=0.01, max_iter=500)), vectorized_data, NN_params)
    train((DecisionTreeClassifier(min_samples_split=60, max_depth=11)), vectorized_data, DT_params)