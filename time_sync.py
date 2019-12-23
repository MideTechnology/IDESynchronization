"""
This code is written for Python 3, and has not been tested (or run).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from fractions import Fraction


def xcorr_norm(x, y):
    """
    The normalized cross-correlation
    """
    return np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


def get_sample_rate(data):
    time_data = data["Time"]
    return len(time_data)/(time_data[-1] - time_data[0])


def get_max_freq(channel, data_index=-1, low_freq=400, high_freq=1500):
    timestep = 1/get_sample_rate(channel)
    channel_data = get_data_index(channel, data_index)
    fft_data = np.abs(np.fft.rfft(channel_data))
    fft_freqs = np.fft.rfftfreq(len(fft_data), d=timestep)
    analysis_freq = fft_freqs[1] - fft_freqs[0]
    low_range = int(low_freq / analysis_freq)
    high_range = int(high_freq / analysis_freq)
    max_freq_index = low_range + np.argmax(fft_data[low_range:high_range])
    max_freq = fft_freqs[max_freq_index]  # USE THIS TO VALIDATE THIS APPROACH AGAINST KNOWN VALUES
    return max_freq


def get_sample_rate_ratio(true_sync, adjust_sync):
    """
    Get the ratio of the two sampling rates for the sync signals (after calculating the adjusted sync signal's
    sampling rate)
    """
    true_sync_freq = get_max_freq(true_sync)
    adjust_sync_freq = get_max_freq(adjust_sync)

    return true_sync_freq / adjust_sync_freq


def apply_time_adjustment(channel, frequency_ratio):
    base = channel["Time"][0]
    channel["Time"] -= base
    channel["Time"] *= frequency_ratio
    channel["Time"] += base


def get_sync_area(signal, sample_frequency):
    points = int(sample_frequency)
    overlap = int(points / 2)
    a, b, c, d = plt.specgram(signal, NFFT=points, noverlap=overlap, Fs=sample_frequency)
    print(a, b, c, d)
    plt.show()


def get_aligned_slices(data1, data2, data1_times, loc1, loc2):
    """
    Align the two data arrays based on the given two locations (having the data at those points line up).
    Both arrays are sliced so that they are the same length.
    """
    num_left_samples = min(loc1, loc2)
    num_right_samples = min(len(data1) - loc1, len(data2) - loc2) - 1

    data1_slice = slice(loc1 - num_left_samples, loc1 + num_right_samples)
    slice1 = data1[data1_slice]
    slice2 = data2[loc2 - num_left_samples: loc2 + num_right_samples]
    time_slice = data1_times[data1_slice]

    return slice1, slice2, time_slice


def resample_slide_and_compare(channel1, channel2, num_points_of_interest=500,
                               similarity_metric=None):
    """
    NOTES:
        - Points of interest are being treated as points with high values, but taking the absolute value prior to
         looking for points of interest may be desired
        - The resampling based on integers may be less than ideal, and is something that should be looked at again when
         there is data to test the code
    """
    if similarity_metric is None:
        similarity_metric = xcorr_norm

    data1 = get_data_index(channel1, -1)
    data2 = get_data_index(channel2, -1)
    data1_times = channel1["Time"]
    samp_rate1 = get_sample_rate(channel1)
    samp_rate2 = get_sample_rate(channel2)

    # Handle resampling with integer approximations of the sampling rate ratio (This can likely be improved)
    samp_rate_ratio = samp_rate1 / samp_rate2
    sample_rate_ratio_approx = Fraction(samp_rate_ratio).limit_denominator(int(1e6))
    samp_rate_1_approx = sample_rate_ratio_approx.numerator
    samp_rate_2_approx = sample_rate_ratio_approx.denominator

    if samp_rate_1_approx < len(data1):
        resampled1 = scipy.signal.resample(data1, samp_rate_1_approx)
    else:
        resampled1 = data1
    if samp_rate_2_approx < len(data2):
        resampled2 = scipy.signal.resample(data2, samp_rate_2_approx)
    else:
        resampled2 = data2

    print("Sample rate ratio approximation error:", samp_rate_ratio - float(sample_rate_ratio_approx))

    # Get the points of interest within the signals
    max_locations1 = np.argpartition(resampled1, -num_points_of_interest)[-num_points_of_interest:]
    max_locations2 = np.argpartition(resampled2, -num_points_of_interest)[-num_points_of_interest:]

    # Check how similar the two signals are when aligning based on two points of interest (one in each signal)
    best_slices = []
    best_score = float('-inf')
    total_locs = len(max_locations1)
    last_ratio = -20
    for c1, loc1 in enumerate(max_locations1):
        if c1/total_locs > last_ratio + 0.10:
            last_ratio = c1/total_locs
            print("%02i Done" % (last_ratio*100))
        for loc2 in max_locations2:
            slices = [slice1, slice2, _] = get_aligned_slices(resampled1, resampled2, data1_times, loc1, loc2)
            new_score = similarity_metric(slice1, slice2)

            if new_score > best_score:
                best_score = new_score
                best_slices = slices

    return best_slices


def get_data_index(numpy_array, index):
    if index >= 0:
        index += 1
    key_name = numpy_array.dtype.names[index]
    return numpy_array[key_name]


def align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync,
                  plot_before_after=True):
    """
    :param true_signal: An ndarray for the signal data which is 'correct'
    :param adjustable_signal: An ndarray for the signal data which should be adjusted
    :param true_sync: The sync signal for the 'correct' signal
    :param adjustable_sync: The sync signal for the adjustable signal
    :param true_time_steps: The times associated with the true_signal data
    :param true_samp_rate: The sampling rate for the 'correct' signal
    :param plot_before_after: If the original and aligned signals should be plotted by matplotlib
    """
    # Calculate the adjustable signal's sampling rate
    sample_rate_ratio = get_sample_rate_ratio(true_sync, adjustable_sync)
    apply_time_adjustment(adjustable_signal, sample_rate_ratio)
    apply_time_adjustment(adjustable_sync, sample_rate_ratio)

    # Align the signals
    aligned = [truth_aligned, adjustable_aligned, aligned_time_steps] = resample_slide_and_compare(
        true_signal,
        adjustable_signal)

    if plot_before_after:
        plt.plot(true_signal["Time"], get_data_index(true_signal, -1))
        plt.plot(adjustable_signal["Time"], get_data_index(adjustable_signal, -1))
        plt.title("Original Signals")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.show()

        plt.plot(aligned_time_steps, truth_aligned)
        plt.plot(aligned_time_steps, adjustable_aligned)
        plt.title("Aligned Signals")
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.show()

    return aligned


if __name__ == "__main__":
    import os.path

    base_dir = "C:/Users/pscheidler/Documents/Slam/Test_Data/A6/SyncTest"
    base_dir = r"C:\_Work\Mide\UTickSynchronization\TestData"
    fns = ["A_Dev_Accel", "A_Dev_Sync", "S_Dev_Accel", "S_Dev_Sync"]
    database = {}
    for fn in fns:
        ffn = os.path.join(base_dir, fn + '.csv')
        npa = np.genfromtxt(ffn, delimiter=',', names=True)
        database[fn.lower()] = npa
    true_signal = database['a_dev_accel']
    true_sync = database['a_dev_sync']
    adj_signal = database['s_dev_accel']
    adj_sync = database['s_dev_sync']
    aligned = align_signals(true_signal, adj_signal, true_sync, adj_sync)
