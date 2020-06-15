# -*- coding: utf-8 -*-
"""
@author: Hugo Moritz
"""
# Improt timer functionality
from timeit import default_timer as timer
from timer_func import print_timer_min_sec

# Import functions
from file_handling import get_file_data
from doc_rep import tfidf
from classification import classification

def main(file):
    """Runs a document classification process on the input file content

    Parameters
    ----------
    file : str
        The file name

    """
    # Start timer
    main_start_timer = timer()
    
    # Get dataframe from file
    df = get_file_data(file)
    
    # Assign t orun with stemmer functionality
    stemmer_func = True
    
    # TF-IDF Document representation functionality
    vectorized_data = tfidf(df)
    
    #Train and test models
    classification(vectorized_data)
    
    # End and print timer
    main_end_timer = timer()
    print('Total time for main():')
    print_timer_min_sec(main_start_timer,main_end_timer)