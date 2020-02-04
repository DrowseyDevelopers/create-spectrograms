#!/usr/bin/env python3
"""
    Module to take in .mat MatLab files and generate spectrogram images via Short Time Fourier Transform
         ----------          ------------------------------          --------------------
        | Data.mat |    ->  | Short-Time Fourier Transform |    ->  | Spectrogram Images |
         ----------          ------------------------------          --------------------
"""

from scipy import signal        # imports to make spectrogram images
import matplotlib.pyplot as plt

import shutil
import numpy as np
import os
import pandas as pd             # data processing
import scipy.io

KEYS = ['id', 'tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']
CWD = os.path.dirname(os.path.realpath(__file__))

DATA_FILES_PATH = os.path.join(CWD, 'data') # constant representing directory path to data files
OUTPUT_PATH = os.path.join(CWD, 'output')   # constant representing directory of generated files
MAT = '.mat'
FREQUENCY = 128                             # frequency rate is 128Hz
M = 128                                     # M = frequency * delta_time = 128 Hz * 15 seconds
MAX_AMP = 2                                 # Max amplitude for short-time fourier transform graph
CHANNELS = [4, 5, 8, 9, 10, 11, 16]


def get_all_data_files():
    """
    Function used to get string values of all files in a directory e.g.
    '/create-spectrograms/data/eeg_record1.mat',
    '/create-spectrograms/data/eeg_record2.mat', etc.
    :return all_files: list of string values of all files in a directory
    """
    all_files = []

    for dirname, _, filenames in os.walk(DATA_FILES_PATH):
        for filename in filenames:

            # ignore anything that is not a .mat file
            if MAT in filename:
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
    raw_data = raw_file['o'][0, 0]

    data = raw_data[6]

    return data


def generate_stft_from_data(channel, fs, m, max_amp, data, output_filepath):
    """
    Function used to generate the Fast-Time Fourier Transform (stft) from data
    :param channel: which channel of the data we are analyzing. Integer value between 0 - 24
    :param fs: frequency sample rate e.g. 128 Hz
    :param m: total number of points in window e.g. 1920
    :param max_amp: max amplitude for stft plot
    :param data: complete dataset from input file
    :param output_filepath: path to export file of short time fourier transform plot of data
    :return None:
    """
    f, t, Zxx = signal.stft(data[:, channel], fs, window='blackman', nperseg=m)

    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=max_amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.savefig(output_filepath)


def generate_spectrogram_from_data(channel, fs, m, data, output_filepath):
    """
    Function used to generate Spectogram images
    :param channel: channel of the data we are analyzing
    :param fs: frequency sample rate e.g. 128 Hz
    :param m: total number of points in window e.g. 128
    :param data: complete dataset from an input file
    :param output_filepath: path to export file of spectrogram
    :return None:
    """
    f, t, Sxx = signal.spectrogram(data[:, channel], fs, window=signal.tukey(m, 0.25))

    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Lead: {0}'.format(str(channel)))
    plt.set_cmap('rainbow')

    plt.savefig(output_filepath)


def create_output_directory(output_path):
    """
    Function used to create the output directory for Short-Time Fourier Transform
    images created for all input files and each channel of an input file.
    if output directory exists, we delete it and recreate it.
    :param output_path: path of the output files we want to create e.g. './output'
    :return None:
    """
    if os.path.isdir(output_path):
        shutil.rmtree(output_path, ignore_errors=True)

    os.mkdir(output_path)


def main():
    """
    Main Entrance of program
    :return None:
    """
    # get all files in input files directory
    all_files = get_all_data_files()

    # create directory where we will output short-time fourier transform images to output to
    create_output_directory(OUTPUT_PATH)

    # iterate through all input data files to generate spectrogram image files
    for data_file in all_files:

        # data from a single file
        data = load_data_from_file(data_file)

        # name of the output image file
        output_basename = os.path.basename(data_file)
        output_basename = output_basename.split('.')[0]

        # full path location of directory we want to create for data file we are analyzing
        output_dirpath = os.path.join(OUTPUT_PATH, output_basename)

        # make a directory for data file being analyzed in order to generate images for all channels of data file.
        # e.g. ./output/eeg_record2/
        os.mkdir(output_dirpath)

        # generating all spectrogram files for all channels of a single EEG data file
        # e.g. ./output/eeg_record2/4.png
        #      ./output/eeg_record2/5.png
        #      ...
        #      ./output/eeg_record2/16.png
        for channel in CHANNELS:
            channel_output_name = '{path}/{channel_index}'.format(path=output_dirpath, channel_index=str(channel))
            generate_spectrogram_from_data(channel, FREQUENCY, M, data, channel_output_name)
            break
        break


if __name__ == '__main__':
    main()

