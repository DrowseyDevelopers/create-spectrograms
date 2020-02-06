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
import scipy.io
import argparse

KEYS = ['id', 'tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']
CWD = os.path.dirname(os.path.realpath(__file__))

# Ranges of data points representing a certain mental state e.g. focused, unfocused or drowsy
FOCUSED_DATA = [0, 76801]
UNFOCUSED_DATA = [76801, 153600]
DROWSEY_DATA = [153601, 230400]

DATA_FILES_PATH = os.path.join(CWD, 'data') # constant representing directory path to data files
STATE_DATA_OUTPUT = os.path.join(CWD, 'state-data')
CHANNELS = [4, 5, 8, 9, 10, 11, 16]

MAT = '.mat'                                # suffix of input files
FREQUENCY = 128                             # frequency rate is 128Hz
M = 128                                     # M = frequency * delta_time = 128 Hz * 15 seconds
MAX_AMP = 2                                 # Max amplitude for short-time fourier transform graph


def handle_arguments():
    """
    Function used to set the arguments that can be passed to the script
    :return: the Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-s', '--split', action='store_true', default=False, dest='split_data',
            help='Flag used to split the data: Focused, Unfocused, and Drowsy data sets')

    return parser.parse_args()


def output_data_to_csv(output_dir, data, state, filename):
    """
    Function used to parse out focused data and output it into csv files
    :param output_dir: directory to output data
    :param data: to output to csv
    :param state: state we are outputting to csv e.g., focused, unfocused or drowsy
    :param filename: name of file we are writing data to
    :return None:
    """

    output_path = os.path.join(output_dir, filename)

    try:
        parsed_data = np.array(data[range(state[0], state[1])])
    except IndexError as e:
        print('File: {0}'.format(output_path))
        print('Size: {0}'.format(len(data)))
        return

    np.savetxt(output_path, parsed_data, delimiter=',')


def handle_split_data(input_files, channels):
    """
    Function used to handle the split of data by mental state
    :return:
    """
    # create directory where we will output split data
    create_output_directory(STATE_DATA_OUTPUT)

    for data_file in input_files:
        # data from a single file
        data = load_data_from_file(data_file)

        # name of the output image file
        output_basename = os.path.basename(data_file)
        output_basename = output_basename.split('.')[0]

        # full path location of directory we want to create for data file we are analyzing
        output_dirpath = os.path.join(STATE_DATA_OUTPUT, output_basename)

        # make a directory for data file being analyzed in order to generate images for all channels of data file.
        # e.g. ./output/eeg_record2/
        os.mkdir(output_dirpath)

        for channel in channels:
            channel_dir = os.path.join(output_dirpath, str(channel))
            os.mkdir(channel_dir)

            output_data_to_csv(channel_dir, data[:, channel], FOCUSED_DATA, 'FOCUSED')
            output_data_to_csv(channel_dir, data[:, channel], UNFOCUSED_DATA, 'UNFOCUSED')
            output_data_to_csv(channel_dir, data[:, channel], DROWSEY_DATA, 'DROWSY')


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
    Function used to generate Spectrogram images
    :param channel: channel of the data we are analyzing
    :param fs: frequency sample rate e.g. 128 Hz
    :param m: total number of points in window e.g. 128
    :param data: complete dataset from an input file
    :param output_filepath: path to export file of spectrogram
    :return None:
    """
    f, t, Sxx = signal.spectrogram(data[0:76800, channel], fs, noverlap=230, window=signal.tukey(256, 0.25))

    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Lead: {0}'.format(str(channel)))
    plt.set_cmap('jet')

    plt.savefig(output_filepath)


def generate_graph_from_data(channel, data, output_filepath):
    """
    Function used to generate time domain graph from channel data
    :param channel: specific channel lead we are analyzing
    :param data: complete dataset from an input file
    :param output_filepath: path to export file of time domain data
    :return None:
    """
    x = np.linspace(0, len(data[:, channel]) / 512., len(data[:, channel]))
    y = data[:, channel]

    plt.plot(x, y, color='blue')
    plt.title('Lead: {}'.format(str(channel)))
    plt.xlabel('Time [secs]')
    plt.ylabel('MicroVolts [muV]')

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
    args = handle_arguments()
    all_files = get_all_data_files()

    if args.split_data:
        handle_split_data(all_files, CHANNELS)


if __name__ == '__main__':
    main()

