# -*- coding: utf-8 -*-
"""
@author: Hugo Moritz
"""

import os
import nltk
import re 

# Import NLTK-library specifics
from nltk.corpus import stopwords as sw
from nltk import wordpunct_tokenize
from nltk import sent_tokenize
from nltk.stem import SnowballStemmer

# To be used with NLTK-library
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/swedish.pickle')

# Initiliaze stemmer
stemmer = SnowballStemmer("swedish")
# Boolean to activate stemmer
STEMMER_FUNC = True

# Get root map
cwd = os.getcwd()

# Read first stopword list
s1 = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\\stopwords-sv.txt'),encoding="utf-8")]
# Read second stopword list
s2 = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\stopwords2-sv.txt'),encoding="utf-8")]
# Read thrid stopword list
s3 = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\\stopwords3-sv.txt'),encoding="utf-8")]
# Read stopword list with cities and municipalities from Sweden
s_cities = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\\stopwords_cities_800-sv.txt'),encoding="utf-8")]
# Read stopword list with common last name from Sweden
s_lastname = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\\stopwords_lastname_100-sv.txt'),encoding="utf-8")]
# Read stopword list with common first name from Sweden
s_firstname = [line.rstrip('\n') for line in open((str(cwd) + '\\stopwords\\stopwords_firstname-sv.txt'))]



def tokenizer(document):
    """Tokenizing a document with pre-processing tasks

    Parameters
    ----------
    document : str
        The document text content.

    Returns
    -------
    list
        A list with the pre-processed tokens.
    """
    
    # Initialize token list
    tokens = []
    
    # Included characters to strip    
    strip_char = '1234567890_–-§:;)(•”�…,.@!#$%^&*()<>?/\⎟|}{~⋅≤='
    
    # Check non-alphabetic characters
    regex_alpha = re.compile('^[a-zA-ZåäöÅÄÖé.]*$')
    
    # Provided stopwords from a few own stopwords added, 
    # the default NLTK English and Swedish stopwords and external stopword lists
    stopwords = {'\n','isbn','isbn:','mtime','pdf','path'}\
    .union(set(sw.words('swedish')),set(sw.words('english')),set(s1),set(s2),set(s3),set(s_cities),set(s_lastname),set(s_firstname))
    
    # Split the document into sentences
    for sent in sent_tokenize(document):
        # Split the document into tokens
        for token in wordpunct_tokenize(sent):
            
            # preprocess and remove unnecessary characters
            token = token.lower()
            token = token.strip(strip_char)
    
            # If still contain special characters, ignore token      
            if(regex_alpha.search(token) == None): 
                continue
            
            # If stopword, ignore token and continue
            if token in stopwords:
                continue
            
            # If token is only one character, ignore token
            if len(token) < 2:
                continue
            
            # Stemmizer functionality
            if STEMMER_FUNC:
                token = stemmer.stem(token)            

            # Add token to the token list
            tokens.append(token)

    return tokens