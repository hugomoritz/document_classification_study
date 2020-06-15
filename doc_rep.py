# -*- coding: utf-8 -*-
"""
@author: Hugo Moritz
"""

# Import document representation specifics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Import functions
from tokenizer import tokenizer

def tfidf(df):
    
    # Setting variables form the dataframe
    '''
        If using different names on columns (using first and second column).
    '''
    #content = df.iloc[:,0]
    #labels = df.iloc[:,1]
    content = df['Content']
    labels = df['Labels']
    

    print('Splitting data...')
    # Split the dataset into training and test datasets 
    x_train, x_test, y_train, y_test = train_test_split(content, labels)
    print('Data splitted.')
        
    # Settings for the tfidf-vectorizer
    '''
        To change to feature selection with Document Frequency (DF), 
        change min_df parameter
    '''
    print('Vectorizing data...')
    tfidf_vec = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,1), min_df=1)
    
    # Executing tf-idf vectorizer
    tfidf_vec.fit(content)
    print('Vectorized data.')
    
    # Tranforming data
    print('Transform x-train and -test...')
    x_train =  tfidf_vec.transform(x_train)
    x_test =  tfidf_vec.transform(x_test)  
    print('Transformed x-train and -test.')
    
    # Number of tokens produced by the vectorizer
    token_dimensionality = len(tfidf_vec.get_feature_names())
    print('Token dimensionality: ', token_dimensionality)

    return x_train, x_test, y_train, y_test