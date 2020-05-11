"""
Author      : Nam Tran, Leonardo Lindo, and Kyle Grace
Class       : HMC CS 158
Date        : 2020 May 12
Description : Soccer Match Winner Predictions
"""

""" !conda update scikit-learn """

# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
# from sklearn.neural_network import MLPClassifier

import csv
from collections import *

import numpy as np
import pandas as pd


# filename = 'data.csv'

# df = pd.read_csv('data.csv', header=0)    # read the file

# print(df)


def transform():
    """ 
    Team names to numbers
    """
    d = { 'Aston Villa':0, 'Wigan':1, 'Blackburn':2, 'Man City':3, 'Bolton':4,
        'Sunderland':5, 'Chelsea':6, 'Hull':7, 'Everton':8, 'Arsenal':9,
        'Portsmouth':10, 'Fulham':11, 'Stoke':12, 'Burnley':13, 'Wolves':14,
        'West Ham': 15, 'Man United': 16, 'Birmingham': 17, 'Tottenham':18, 'Liverpool':19,
        'West Brom':20, 'Newcastle':21, 'Blackpool':22, 'QPR': 23, 'Swansea':24,
        'Norwich':25, 'Reading': 26, 'Southampton': 27, 'Crystal Palace': 28,
        'Cardiff':29, 'Leicester':30, 'Bournemouth':31, 'Watford':32, 'Middlesbrough':33,
        'Brighton':34, 'Huddersfield':35}
    return d