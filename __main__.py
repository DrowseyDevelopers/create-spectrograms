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

DATA_FILES_PATH = os.path.join(CWD, 'data') # constant representing directory path to data files
FREQUENCY = 128                             # frequency rate is 128Hz
M = 1920                                    # M = frequency * delta_time = 128 Hz * 15 seconds
MAX_AMP = 2                                 # Max amplitude for short-time fourier transform graph

"""
data is a 25x(# ofDataPoints)

we want rows 4-17
thus (4-17)x(:)
"""


def get_all_data_files():
    """
    Function used to get string values of all files in a directory e.g.
    /create-spectrograms/data/eeg_record1.mat,
    /create-spectrograms/data/eeg_record2.mat, etc.
    :return all_files: list of string values of all files in a directory
    """
    all_files = []

    for dirname, _, filenames in os.walk(DATA_FILES_PATH):
        for filename in filenames:
            # Example: complete_path_to_file = /create-spectrograms/data/eeg_record1.mat
            complete_path_to_file = os.path.join(dirname, filename)
            all_files.append(complete_path_to_file)

    return all_files


def load_data_from_file(path_to_file):
    """
    Function used to get data from a .mat file
    :param path_to_file: path to file we want to read e.g. /create-spectrograms/data/eeg_record2.mat
    :return data: numpy 2-D array 25x308868 to represent all data points gathered in 25 channels
    """
    raw_file = scipy.io.loadmat(path_to_file)
    raw_data = pd.DataFrame.from_dict(raw_file['o']['data'][0, 0])

    data = pd.DataFrame(raw_data).to_numpy()

    return data


def generate_stft_from_data(channel, fs, m, max_amp, data):
    """
    Function used to generate the Fast-Time Fourier Transform (stft) from data
    :param channel: which channel of the data we are analyzing. Integer value between 0 - 24
    :param fs: frequency sample rate e.g. 128 Hz
    :param m: total number of points in window e.g. 1920
    :param max_amp: max amplitude for stft plot
    :param data: complete dataset from input file
    :return None:
    """
    f, t, Zxx = signal.stft(data[:, channel], fs, window='blackman', nperseg=m)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=max_amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def main():
    """
    Main Entrance of program
    :return None:
    """

    all_files = get_all_data_files()

    for data_file in all_files:

        data = load_data_from_file(data_file)
        generate_stft_from_data(3, FREQUENCY, M, MAX_AMP, data)
        print(data_file)
        break

    # time variable = [1, 2, ... 308868]
    time = [item for item in range(1, np.size(data, 0) + 1)]


if __name__ == '__main__':
    main()

