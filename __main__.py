#!/usr/bin/env python3

# imports to make spectrogram images
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.io

KEYS = ['id', 'tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']

CWD = os.path.dirname(os.path.realpath(__file__))

# constant representing directory path to data files
DATA_FILES_PATH = os.path.join(CWD, 'data')

"""
data is a 25x(# ofDataPoints)

we want rows 4-17
thus (4-17)x(:)
"""


def get_data_files():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def main():
    """
    Main Entrance of program
    :return None:
    """
    print(CWD)

    raw_file = scipy.io.loadmat('{0}/data/eeg_record30.mat'.format(CWD))
    obj = raw_file['o']

    data_unclean = pd.DataFrame.from_dict(raw_file["o"]["data"][0,0])
    data = pd.DataFrame(data_unclean).to_numpy()

    fs = 128
    amp = 2
    f, t, Zxx = signal.stft(data[:, 3], fs, window='blackman', nperseg=1920)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # time variable = [1, 2, ... 308868]
    time = [item for item in range(1, np.size(data, 0) + 1)]


if __name__ == '__main__':
    main()
