#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.io
KEYS = ['id', 'tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']

def main():
    """
    Main Entrance of program
    :return None:
    """
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)

    raw_file = scipy.io.loadmat('EEGData/eeg_record1.mat')
    obj = raw_file['o']

    data = pd.DataFrame.from_dict(raw_file["o"]["data"][0,0])

    if data:
        print('yay')

if __name__ == '__main__':
    main()
