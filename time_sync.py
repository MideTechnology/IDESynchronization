from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import numba as nb
import matplotlib
matplotlib.use('wxAgg')
import matplotlib.pyplot as plt
import scipy.signal
import os
import csv
import sys
from fractions import Fraction

from IDESynchronization.ide_csv_conversion.Ide2CsvWrapper import Ide2CsvWrapper
import IDESynchronization.ide_csv_conversion.ide_helpers as ide_helpers


@nb.njit
def xcorr_norm(x, y):
    """
    The normalized cross-correlation
    """
    total_x = 0
    total_y = 0
    for j in range(len(x)):
        total_x += x[j]
        total_y += y[j]
    x_mean = total_x / len(x)
    y_mean = total_y / len(y)

    numerator_sum = 0
    x_sum = 0
    y_sum = 0
    for j in range(len(x)):
        numerator_sum += (x[j] - x_mean) * (y[j] - y_mean)
        x_sum += x[j] ** 2
        y_sum += y[j] ** 2

    return numerator_sum / np.sqrt(x_sum * y_sum)

    # The below code should be used for a non-compiled implementation
    # return np.sum((x-np.mean(x))*(y-np.mean(y))) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


@nb.njit
def get_aligned_slices(true_signal, adj_signal, true_signal_times, true_loc, adj_loc):
    """
    Align the two data arrays based on the given two locations (having the data at those points line up).
    Both arrays are sliced so that they are the same length.

    :param true_signal: An ndarray for the resampled signal data which is 'correct'
    :param adj_signal: An ndarray for the resampled signal data which should be adjusted
    :param true_signal_times: The time data associated with the given true_signal
    :param true_loc: The location in the 'true' signal to align with the given location in the 'adjustable' signal
    :param adj_loc: The location in the 'adjustable' signal to align with the given location in the 'true' signal
    :return: A length three tuple where the first and second element are the true and adjustable signals sliced so that
     every point in one corresponds to a point in the other, and the third element is the time's sliced to correspond to
     the true signal
    """
    num_left_samples = min(true_loc, adj_loc)
    num_right_samples = min(len(true_signal) - true_loc, len(adj_signal) - adj_loc) - 1

    true_signal_slice = slice(true_loc - num_left_samples, true_loc + num_right_samples)
    slice1 = true_signal[true_signal_slice]
    slice2 = adj_signal[adj_loc - num_left_samples: adj_loc + num_right_samples]
    time_slice = true_signal_times[true_signal_slice]

    return slice1, slice2, time_slice


def get_sample_period(timestamps):
    return (timestamps[-1] - timestamps[0])/(len(timestamps)-1)


def get_max_freq(channel_data, sample_period, low_freq_range=400, high_freq_range=1500):
    fft_data = np.abs(np.fft.rfft(channel_data))
    fft_freqs = np.fft.rfftfreq(len(channel_data), d=sample_period)
    analysis_freq = fft_freqs[1] - fft_freqs[0]
    low_range = int(low_freq_range / analysis_freq)
    high_range = int(high_freq_range / analysis_freq)
    max_freq_index = low_range + np.argmax(fft_data[low_range:high_range])
    max_freq = fft_freqs[max_freq_index]  # USE THIS TO VALIDATE THIS APPROACH AGAINST KNOWN VALUES
    return max_freq


def get_sample_rate_ratio(true_sync, adjust_sync, true_timestep, adjust_timestep, plot_info=False):
    """
    Get the ratio of the two sampling rates for the sync signals (after calculating the adjusted sync signal's
    sampling rate)

    :param true_sync: The sync data for the 'true' signal
    :param adjust_sync: The sync data for the 'adjustable' signal
    :param true_timestep: The timestep for the 'true' signal
    :param adjust_timestep: The timestep for the 'adjustable' signal
    :param plot_info: A boolean indicating if the true and adjustable signal's FFTs should be plotted with matplotlib
    :return: The ratio of the sampling rates between the true and adjustable signals
    """
    true_sync_freq = get_max_freq(true_sync, true_timestep)
    adjust_sync_freq = get_max_freq(adjust_sync, adjust_timestep)
    return true_sync_freq / adjust_sync_freq


def resample_slide_and_compare(true_signal, adj_signal, true_signal_times, samp_rate1, samp_rate2, max_start_offset,
                               progress_callback, similarity_metric=None, plot_info=False):
    """

    :param true_signal: An ndarray for the signal data which is 'correct'
    :param adj_signal: An ndarray for the signal data which should be adjusted
    :param true_signal_times: The time data associated with the given true_signal
    :param samp_rate1: The sampling rate of the first signal
    :param samp_rate2: The sampling rate of the second signal
    :param max_start_offset: The maximum starting time difference to be checked allowed when trying to sync the signals
    :param similarity_metric: A function to be used as the similarity metric, or None which defaults to using the
     normalized cross correlation.  The function should take in two ndarrays (the two signals to compare), and output a
     NumPy float32, which should be higher the more similar the two signals are.
    :param plot_info: A boolean value indicating if the resampled data and it's points of interest should be plotted
     by matplotlib
    :return: The same as the return from the allign_location_search function (see it's docstring)

    NOTES:
     - The resample function calls are EXTREMELY slow if resampling to a prime number of samlpes (like so slow it'll
      just appear to hang)
    """
    MAX_SAMP_RATE_RATIO_DENOMINATOR = int(1e6)
    MIN_LENGTH_MULTIPLIER = 10

    if similarity_metric is None:
        similarity_metric = xcorr_norm

    # Handle resampling with integer approximations of the sampling rate ratio (This can likely be improved)
    samp_rate_ratio = samp_rate1 / samp_rate2 * len(true_signal) / len(adj_signal)

    sample_rate_ratio_approx = Fraction(samp_rate_ratio)
    if samp_rate_ratio > 1:
        sample_rate_ratio_approx = 1/((1/sample_rate_ratio_approx).limit_denominator(MAX_SAMP_RATE_RATIO_DENOMINATOR))
    else:
        sample_rate_ratio_approx = sample_rate_ratio_approx.limit_denominator(MAX_SAMP_RATE_RATIO_DENOMINATOR)

    if min(sample_rate_ratio_approx.numerator, sample_rate_ratio_approx.denominator) < MIN_LENGTH_MULTIPLIER * max(len(true_signal), len(adj_signal)):
        scaler_multiplier = int(max(sample_rate_ratio_approx.numerator, sample_rate_ratio_approx.denominator) / (MIN_LENGTH_MULTIPLIER * max(len(true_signal), len(adj_signal))))
        scaler_multiplier = max(1, scaler_multiplier)
    else:
        scaler_multiplier = 1

    samp_rate_1_approx = sample_rate_ratio_approx.numerator * scaler_multiplier
    samp_rate_2_approx = sample_rate_ratio_approx.denominator * scaler_multiplier

    progress_callback("Resampling the signals for consistent frequency")

    # resampled1 = scipy.signal.resample(true_signal, samp_rate_1_approx)
    # resampled2 = scipy.signal.resample(adj_signal, samp_rate_2_approx)
    resampled1 = scipy.signal.resample_poly(true_signal, samp_rate_1_approx, len(true_signal))#, padtype="median")
    resampled2 = scipy.signal.resample_poly(adj_signal, samp_rate_2_approx, len(adj_signal))#, padtype="median")


    # print("Sample rate ratio approximation error:", samp_rate_ratio - float(sample_rate_ratio_approx))

    progress_callback("Finding POI (peaks and valleys)")

    # Get the points of interest within the signals
    peaks1 = scipy.signal.find_peaks(resampled1, prominence=1)[0] ############ SHOULD LOOK INTO USING find_peaks_cwt #############
    peaks2 = scipy.signal.find_peaks(resampled2, prominence=1)[0]
    valleys1 = scipy.signal.find_peaks(-resampled1, prominence=1)[0]
    valleys2 = scipy.signal.find_peaks(-resampled2, prominence=1)[0]

    # Combine the peaks with the valleys for each signal
    poi1 = np.union1d(peaks1, valleys1)
    poi2 = np.union1d(peaks2, valleys2)

    # Get the timestamps associated with resampled1
    # true_signal_times = np.linspace(true_signal_times[0], true_signal_times[-1], len(resampled1)+2)[1:-1]
    true_signal_times = np.linspace(true_signal_times[0], true_signal_times[-1], len(resampled1))

    # if plot_info:
    #     print("%d points of interest in signal 1\n%d points of intereste in signal 2"%(len(poi1), len(poi2)))
    #     time_step = (true_signal_times[-1] - true_signal_times[0])/(len(resampled1) - 1)
    #
    #     plt.plot(resampled1)
    #     plt.plot(resampled2)
    #     plt.plot(poi1, resampled1[poi1], "x")
    #     plt.plot(poi2, resampled2[poi2], 'x')
    #     plt.xlabel("Time Steps (%.5e seconds)" % time_step)
    #     plt.title("Resampled Data with Marked POI ")
    #     plt.show()

    progress_callback("Aligning POI pairs for optimal time offset")

    return allign_location_search(resampled1, resampled2, true_signal_times, poi1, poi2, similarity_metric, max_start_offset)

@nb.njit
def allign_location_search(true_signal, adj_signal, true_signal_times, poi1, poi2, similarity_metric, max_start_offset):
    """
    Check how similar the two signals are when aligning based on two points of interest (one in each signal), and
    record the 'most' aligned configuration.

    :param true_signal: An ndarray for the resampled signal data which is 'correct'
    :param adj_signal: An ndarray for the resampled signal data which should be adjusted
    :param true_signal_times: The time data associated with the given true_signal
    :param poi1: The points of interest in the 'true' signal to use when lining up the signals
    :param poi2: The points of interest in the 'adjustable' signal to use when lining up the signals
    :param similarity_metric: A function to be used as the similarity metric, or None which defaults to using the
     normalized cross correlation.  The function should take in two ndarrays (the two signals to compare), and output a
     NumPy float32, which should be higher the more similar the two signals are.
    :param max_start_offset: The maximum starting time difference to be checked allowed when trying to sync the signals
    :return: A tuple of the same form as the return of the get_aligned_slices function (see it's docstring),
     corresponding to the most aligned slices of data
    """
    time_increment = (true_signal_times[-1] - true_signal_times[0]) / (len(true_signal_times) - 1) ####DOUBLE CHECK THE DENOMINATOR######

    best_slices = (None, None, None, None, None)
    best_score = np.finfo(np.float32).min
    for loc1 in poi1:
        for loc2 in poi2:
            if abs(loc1 - loc2) * time_increment <= max_start_offset:
                slices = [slice1, slice2, _] = get_aligned_slices(true_signal, adj_signal, true_signal_times, loc1, loc2)
                new_score = similarity_metric(slice1, slice2)

                if new_score > best_score:
                    best_score = new_score
                    best_slices = slices + (loc1, loc2)

    return best_slices


def align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync, true_time_signal, adjustable_time_signal,
                  true_time_sync, adjustable_time_sync, progress_callback, max_start_offset=None, plot_info=False):
    """
    TODO:
     - Plot against time rather than index number

    :param true_signal: An ndarray for the signal data which is 'correct'
    :param adjustable_signal: An ndarray for the signal data which should be adjusted
    :param true_sync: The sync signal for the 'correct' signal
    :param adjustable_sync: The sync signal for the adjustable signal
    :param true_time_signal: The times associated with the true_signal data
    :param adjustable_time_signal: The times associated with the adjustable_signal data
    :param true_sample_period: The sampling rate for the 'correct' signal
    :param max_start_offset: The maximum starting time difference to be checked allowed when trying to sync the signals
    :param plot_info: If the original and aligned signals should be plotted by matplotlib
    """
    # if plot_info:
    #     plt.plot(true_signal)
    #     plt.plot(adjustable_signal)
    #     plt.title("Original Signals")
    #     plt.xlabel("Time Steps")
    #     plt.show()
    #
    #     plt.plot(true_time_steps, true_signal)
    #     plt.plot(adjustable_time_steps, adjustable_signal)
    #     plt.title("Original Signals")
    #     plt.xlabel("Time (s)")
    #     plt.show()

    # If the maximum start offset is not given, set it to one fourth of the true signal's length
    if max_start_offset is None:
        max_start_offset = max(true_time_signal[-1] - true_time_signal[0], adjustable_time_signal[-1] - adjustable_time_signal[0]) / 4

    true_time_increment = get_sample_period(true_time_signal)
    adjustable_time_increment = get_sample_period(adjustable_time_signal)
    true_sync_period = get_sample_period(true_time_sync)
    adjustable_sync_period = get_sample_period(adjustable_time_sync)

    progress_callback("Computing sampling frequency error")

    # Calculate the adjustable signal's sampling rate
    sample_rate_ratio = get_sample_rate_ratio(true_sync, adjustable_sync, true_sync_period,
                                              adjustable_sync_period, plot_info)

    actual_adj_sampling_period = adjustable_time_increment / sample_rate_ratio

    # Align the signals
    aligned = [truth_aligned, adjustable_aligned, aligned_time_steps, poi1, poi2] = resample_slide_and_compare(
        true_signal,
        adjustable_signal,
        true_time_signal,
        true_time_increment,
        actual_adj_sampling_period,
        progress_callback=progress_callback,
        max_start_offset=max_start_offset,
        plot_info=plot_info)

    resampled_sample_period = (aligned_time_steps[-1] - aligned_time_steps[0]) / (len(aligned_time_steps) - 1)
    adj_start_time = aligned_time_steps[0] + (poi1 - poi2) * resampled_sample_period

    adj_times_fixed = adj_start_time + actual_adj_sampling_period * np.arange(len(adjustable_time_signal))

    if plot_info:
        # The below commented out code plots the resampled data with it's resampled time stamps
        fig, (ax1, ax2) = plt.subplots(2, num="Synchronization Results %d" % (1+max([0] + plt.get_fignums())))
        ax1.plot(true_time_signal, true_signal, label="True Signal")
        ax1.plot(adjustable_time_signal, adjustable_signal, label="Adjustable Signal")
        ax1.set_title("Original Data")
        ax2.plot(true_time_signal, true_signal, label="True Signal")
        ax2.plot(adj_times_fixed, adjustable_signal, label="Adjustable Signal")
        ax2.set_title("Synchronized Data")
        # fig.suptitle("Before and After Synchronization")
        ax1.set(xlabel='Time (s)')
        ax2.set(xlabel='Time (s)')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show(block=False)

    return aligned, adj_times_fixed, sample_rate_ratio


def load_csv_data(base_dir):
    """
    TODO:
     - Have this take in file path+name for all the files, rather than the base directory
     - Remove this function entirely

    :return: A dictionary mapping a signals string identification to the signal's data
    """
    filenames = ["A_Dev_Accel", "A_Dev_Sync", "S_Dev_Accel", "S_Dev_Sync"]
    database = {}
    for name in filenames:
        ffn = os.path.join(base_dir, name + '.csv')
        npa = np.genfromtxt(ffn, delimiter=',', skip_header=1)
        database[name.lower()] = {'time': npa[:, 0], 'data': npa[:, -1]}

    return {
        "true_signal": database['a_dev_accel']['data'],
        "true_sync": database['a_dev_sync']['data'],
        "adj_signal": database['s_dev_accel']['data'],
        "adj_sync": database['s_dev_sync']['data'],
        "true_signal_time": database['a_dev_accel']['time'],
        "adj_signal_time": database['s_dev_accel']['time'],
        "true_sync_time": database['a_dev_sync']['time'],
        "adj_sync_time": database['s_dev_sync']['time'],
    }

def new_load_csv_data(filename_dict):
    """
    TODO:
     - Have this take in file path+name for all the files, rather than the base directory

    :return: A dictionary mapping a signals string identification to the signal's data
    """
    database = {}
    for name in filename_dict.values():
        npa = np.genfromtxt(name, delimiter=',', skip_header=1)
        with open(name, 'r') as f:
            has_sync_mask = np.array(list(map(lambda x: "Sync" in x, f.readline().split(','))))
            nonzero_indecies = has_sync_mask.nonzero()[0]
            if len(nonzero_indecies) == 0:  # If the data is acceleration data
                # Get the magnitude of the x, y, and z accelrations
                database[name] = {'time': npa[:, 0], 'data': np.sqrt(np.sum(npa[:, 1:]**2, axis=1))}
            else:
                data_column = nonzero_indecies[0]
                database[name] = {'time': npa[:, 0], 'data': npa[:, data_column]}

    return {
        "true_signal": database[filename_dict['true_signal']]['data'],
        "true_sync": database[filename_dict['true_sync']]['data'],
        "adj_signal": database[filename_dict['adj_signal']]['data'],
        "adj_sync": database[filename_dict['adj_sync']]['data'],
        "true_signal_time": database[filename_dict['true_signal']]['time'],
        "adj_signal_time": database[filename_dict['adj_signal']]['time'],
        "true_sync_time": database[filename_dict['true_sync']]['time'],
        "adj_sync_time": database[filename_dict['adj_sync']]['time'],
    }


def sync_and_create_new_csv(true_ide_path, adj_ide_path, output_dir, convert_all_channels=True, progress_callback=None, show_signal_plots=False):
    if progress_callback is None:
        progress_callback = lambda x: None

    to_convert_to_csv = [true_ide_path, adj_ide_path]
    conversion_executable = "ide_csv_conversion\\ide2csv_64b.exe"
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    conversion_executable = os.path.join(bundle_dir, conversion_executable)

    progress_callback("Converting IDE files to CSV")

    # Create csv files for the IDE file
    channels_to_convert = ide_helpers.channels_by_name.keys() if convert_all_channels else [8, 80]
    ide_to_csv_converter = Ide2CsvWrapper(to_convert_to_csv, channels=channels_to_convert,
                                          converter=conversion_executable, output_path=output_dir)
    ide_to_csv_converter.run()

    progress_callback("Loading CSV data")

    split_true_ide_name = true_ide_path.split('\\')
    split_adj_ide_name = adj_ide_path.split('\\')

    true_ide_name = split_true_ide_name[-1].split('.')[0]
    adj_ide_name = split_adj_ide_name[-1].split('.')[0]

    filename_dict = {
        "true_signal": "%s\\%s_Ch80.csv" % (output_dir, true_ide_name),
        "true_sync": "%s\\%s_Ch08.csv" % (output_dir, true_ide_name),
        "adj_signal": "%s\\%s_Ch80.csv" % (output_dir, adj_ide_name),
        "adj_sync": "%s\\%s_Ch08.csv" % (output_dir, adj_ide_name),
    }

    # Load csv data
    data_dict = new_load_csv_data(filename_dict)

    # Get synchronized timesteps

    _, new_adj_times, sample_rate_ratio = align_signals(
        data_dict['true_signal'],   # true_signal
        data_dict['adj_signal'],    # adjustable_signal
        data_dict['true_sync'],     # true_sync
        data_dict['adj_sync'],      # adjustable_sync
        data_dict['true_signal_time'],     # true_time_steps
        data_dict['adj_signal_time'],      # adjustable_time_steps
        data_dict['true_sync_time'],     # true_time_steps
        data_dict['adj_sync_time'],      # adjustable_time_steps
        progress_callback=progress_callback,
        plot_info=show_signal_plots,
    )

    progress_callback("Creating adjusted CSV files")

    time_offset = new_adj_times[0] - data_dict["adj_signal_time"][0]

    # build list of adjustable CSV files
    adj_files = [fn for fn in os.listdir(output_dir) if fn.endswith('.csv') and fn.startswith(adj_ide_name) and
                 not fn.endswith("adjusted.csv")]

    for fn in adj_files:
        with open(os.path.join(output_dir, fn)) as f:
            reader = csv.reader(f)
            new_signal_data = np.array(list(reader))
#		new_signal_data[1:, 0] = new_signal_data[1:, 0].astype(np.float)
        start = np.float(new_signal_data[1, 0]) + time_offset
        adjusted_times = new_signal_data[1:, 0].astype(np.float) - np.float(new_signal_data[1, 0])
        adjusted_times /= sample_rate_ratio
        adjusted_times += start
        new_signal_data[1:, 0] = adjusted_times

        new_csv_filename = "%s//%s_adjusted.csv" % (output_dir, fn[:-4])
        with open(new_csv_filename, 'wb') as f:		# Note: Change to wb for Python2, w for Python3. Python3 also needs to remove \n
            writer = csv.writer(f)
            writer.writerows(new_signal_data)

