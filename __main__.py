#!/usr/bin/env python3
"""
    Module to take in .mat MatLab files and generate spectrogram images via Short Time Fourier Transform
         ----------          ------------------------------          --------------------
        | Data.mat |    ->  | Short-Time Fourier Transform |    ->  | Spectrogram Images |
         ----------          ------------------------------          --------------------
"""

from scipy import signal  # imports to make spectrogram images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import numpy as np
import os
import scipy.io
import argparse
import glob
import math

np.seterr(divide='raise')

KEYS = ['id', 'tag', 'nS', 'sampFreq', 'marker', 'timestamp', 'data', 'trials']
CWD = os.path.dirname(os.path.realpath(__file__))

# Ranges of data points representing a certain mental state e.g. focused, unfocused or drowsy
FOCUSED_DATA = [0, 76801]
UNFOCUSED_DATA = [76801, 153600]
DROWSY_DATA = [153601, 230400]

DATA_FILES_PATH = os.path.join(CWD, 'data')  # constant representing directory path to data files
STATE_DATA_OUTPUT = os.path.join(CWD, 'state-data')
CHANNELS = [4, 5, 8, 9, 10, 11, 16]

MAT = '.mat'  # suffix of input files
FREQUENCY = 128  # frequency rate is 128Hz
M = 64
MAX_AMP = 2  # Max amplitude for short-time fourier transform graph


def handle_arguments():
    """
    Function used to set the arguments that can be passed to the script
    :return: the Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-s', '--split', action='store_true', default=False, dest='split_data',
                        help='Flag used to split the data: Focused, Unfocused, and Drowsy data sets')

    parser.add_argument('-i', '--images', dest='state', choices=['FOCUSED', 'UNFOCUSED', 'DROWSY', 'ALL'],
                        help='Flag used to determine what mental state we want to create spectrogram images for')

    return parser.parse_args()


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
            output_data_to_csv(channel_dir, data[:, channel], DROWSY_DATA, 'DROWSY')


def handle_create_spectrograms(state):
    """
    Function used to determine what what state (e.g., FOCUSED, UNFOCUSED, DROWSY, or ALL) spectrogram
    images to create
    :param state:
    :return None:
    """
    states = []

    if state == 'ALL':
        states = ['FOCUSED', 'UNFOCUSED', 'DROWSY']
    else:
        states = [state]

    # need to check if state-data directory exists in path
    if not os.path.isdir(STATE_DATA_OUTPUT):
        print('Error: Directory \'{0}\' with raw input data doesnt exists!'.format(STATE_DATA_OUTPUT))
        exit(1)

    # iterate through states that we need to generate spectrogram images for
    for curr_state in states:
        output_root = os.path.join(CWD, curr_state)

        create_output_directory(output_root)

        path_to_search = os.path.join(STATE_DATA_OUTPUT, '**', curr_state)
        state_data_files = glob.glob(path_to_search, recursive=True)

        for filename in state_data_files:
            output_subpath = filename.replace(STATE_DATA_OUTPUT, '')
            output_subpath = output_subpath.replace(curr_state, '')
            output_filepath = '{0}{1}'.format(output_root, output_subpath)

            os.makedirs(output_filepath)

            # need to get data from file
            data = load_raw_state_data(filename)

            output_image = os.path.join(output_filepath, curr_state)

            # 128, 256, 10mins, ./FOCUSED/eeg_record7/10/FOCUSED
            interate_data(FREQUENCY, M, data, output_image)


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


def load_raw_state_data(path_to_file):
    """
    Function to load raw state data from a csv file
    :param path_to_file: the path to file we want to read
    :return data: raw data from file
    """
    data = np.genfromtxt(path_to_file)

    return data


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


def generate_spectrogram_from_data(fs, m, data, output_filepath):
    """
    Function used to generate Spectrogram images
    :param fs: frequency sample rate e.g. 128 Hz
    :param m: total number of points in window e.g. 128
    :param data: complete dataset from an input file
    :param output_filepath: path to export file of spectrogram
    :return None:
    """
    overlap = math.floor(m * 0.9)

    f, t, Sxx = signal.spectrogram(data, fs, noverlap=overlap, window=signal.tukey(m, 0.25))

    try:
        plt.pcolormesh(t, f, np.log10(Sxx))
        plt.set_cmap('jet')
        plt.axis('off')

        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, dpi=35)
        plt.clf()
    except FloatingPointError as e:
        print('Caught divide by 0 error: {0}'.format(output_filepath))
        return


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


def interate_data(fs, m, data, output_file):
    """
    Function used to interate through data and generate spectrogram images
    :param fs:
    :param m:
    :param data:
    :param output_file:
    :return:
    """
    move = 128
    i = 0
    j = 256
    counter = 1

    while j < len(data):
        sub_data = data[i:j]

        # FOCUSED/eeg_record7/10/FOCUSED_1
        sub_output_file = '{0}_{1}'.format(output_file, str(counter))

        generate_spectrogram_from_data(fs, m, sub_data, sub_output_file)

        i += move
        j += move
        counter += 1


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


def main():
    """
    Main Entrance of program
    :return None:
    """
    args = handle_arguments()

    all_files = get_all_data_files()

    if args.split_data:
        handle_split_data(all_files, CHANNELS)

    if args.state:
        handle_create_spectrograms(args.state)


if __name__ == '__main__':
    main()
