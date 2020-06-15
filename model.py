# -*- coding: utf-8 -*-
"""
@author: Hugo Moritz
"""

# Import model specifics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Import evaluation specifics
from sklearn.metrics import classification_report
from sklearn import metrics

# Import warnings
import warnings
 
# Import timer functionality
from timeit import default_timer as timer
from timer_func import print_timer_min_sec

def train(model, vectorized_data, params = None):
    """Trains and tests the a machine learning model and shows the result.

    Parameters
    ----------
    model: estimator
        Machine Learning model
    vectorized_data: numpy.Arrays
        x_train, x_test, y_train, y_test rf-idf vectorized variables
    params: Dictionary
        Optimising parameters

    """
    #Removing warnings
    warnings.filterwarnings('ignore') 

    # Getting the x and y train/test variables
    x_train, x_test, y_train, y_test = vectorized_data

    # Starting training timer
    training_start_timer = timer()
    
    # Getting model name
    model_name = type(model).__name__
    
    # Printing 
    print(' ')
    print('Training model..')

    ''' If parameters for optimisation is assigned, optimisation
        is used.
    '''
    if params != None:    
        clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
        clf.fit(x_train, y_train)
        best_params = clf.best_params_
        print('Best params:    ', best_params)
        print('Best estimator: ', clf.best_estimator_)
        print('Best score:     ', clf.best_score_)
    else:
        # Pipelining.
        # Possible to uncomment feature selection.
        clf = Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=False))),
            ('classification', model)])
        clf.fit(x_train, y_train)
    
    print('Model trained.')
    
    # End training timer
    training_end_timer = timer()
    
    print('Testing ', model_name, '...')
    # Start test timer
    test_start_timer = timer()
    
    # Get accuracy score for training and testing
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    
    # Predicting the score paramters
    y_pred = clf.predict(x_test)
    
    # End test timer
    test_end_timer = timer()    
    
    print('Model tested.')
    print(' ')
    print('-----------------------------------------------')
    print(model_name)
    print(' ')
    # Print the score
    print("Train accuracy: ", train_score)
    print("Test accuracy:  ", test_score)

    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(model_name, clf, x_test, y_test)
    
    # Print training and testing time
    print(' ')
    print('Training time: ')
    print_timer_min_sec(training_start_timer,training_end_timer)
    print(' ')
    print('Test time: ')
    print_timer_min_sec(test_start_timer,test_end_timer)
    print('-----------------------------------------------')
    
    return None

def plot_confusion_matrix(model_name, clf, x_test, y_test):
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")