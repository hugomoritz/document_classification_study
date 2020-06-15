# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:42:19 2020

@author: Hugo Moritz
"""
import pandas as pd


def get_file_data(file):
    """Reads the file to a the dataframe

    Parameters
    ----------
    file : str
        The file name

    Returns
    -------
    pd.DataFrame
        A dataframe with the content of the file
    """
    print('Reading dataframe file..')
    df = pd.read_csv(file) 
    print('Read dataframe file.')
    return df