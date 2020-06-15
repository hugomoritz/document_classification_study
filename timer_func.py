# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:11:10 2020

@author: Hugo Moritz
"""

def truncate(n, decimals=0):
    """Truncates n

    Parameters
    ----------
    n : float
        Numerical value
    
    decimals : int
        Number of decimals for returned value

    Returns
    -------
    int
        Truncated integer
    """
    multiplier = 10 ** decimals
    return int(int(n * multiplier) / multiplier)

def convert_timer_min_sec(start, end):
    """Calculates start and end timestamp difference in minutes and seconds

    Parameters
    ----------
    start : string
        Start timestamp
    
    end : string
        End timestamp

    Returns
    -------
    int
        The difference of the timestamps for the minutes
        
    int
        Difference of the timestamps for the seconds
    """
    minutes = truncate((end-start)/60)
    seconds = (((end-start)/60)-minutes)*60
    return minutes, seconds
    
def print_timer_min_sec(start, end):
    """Prints time in minutes and seconds of start and end timestamp 
    differences.

    Parameters
    ----------
    start : string
        Start timestamp
    
    end : string
        End timestamp
    """
    minutes, seconds = convert_timer_min_sec(start, end)
    print('\n','Time: ', minutes, 'm ', seconds, 's')
    return None

