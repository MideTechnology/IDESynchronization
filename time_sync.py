"""
This code is written for Python 3.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.signal
import os
from fractions import Fraction


@nb.njit
def xcorr_norm(x, y):
	"""
	The normalized cross-correlation
	"""
	total_sum = 0
	x_sum = 0
	y_sum = 0
	for j in range(len(x)):
		total_sum += x[j] * y[j]
		x_sum += x[j] ** 2
		y_sum += y[j] ** 2
	return total_sum / np.sqrt(x_sum * y_sum)

	# The below code should be used for a non-compiled implementation
	# return np.sum(x*y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


def get_sample_rate_ratio(true_sync, adjust_sync, true_timestep, plot_info=False):
	"""
	Get the ratio of the two sampling rates for the sync signals (after calculating the adjusted sync signal's
	sampling rate)

	TODO:
	 - Plot against frequencies rather than index number

	:param true_sync: The sync data for the 'true' signal
	:param adjust_sync: The sync data for the 'adjustable' signal
	:param true_timestep: The timestep for the 'true' signal
	:param plot_info: A boolean indicating if the true and adjustable signal's FFTs should be plotted with matplotlib
	:return: The ratio of the sampling rates between the true and adjustable signals
	"""
	true_fft = np.abs(np.fft.rfft(true_sync))
	rfft_length = len(true_sync)  # len(true_fft) # Double check this
	true_fft_freqs = np.fft.rfftfreq(rfft_length, d=true_timestep)

	adjust_fft = np.abs(np.fft.rfft(adjust_sync))


	# This is seems to fix an issue that was being run into which I don't fully understand
	true_fft[0] = adjust_fft[0] = np.finfo(adjust_fft.dtype).min

	true_sync_freq = true_fft_freqs[np.argmax(true_fft)]

	if plot_info:
		plt.plot(true_fft[1:])
		plt.plot(adjust_fft[1:])
		plt.title("Sync Signal's FFT")
		plt.show()

	adjust_sync_freq = true_fft_freqs[np.argmax(adjust_fft)]
	return true_sync_freq / adjust_sync_freq

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


def resample_slide_and_compare(true_signal, adj_signal, true_signal_times, samp_rate1, samp_rate2, max_start_offset,
							   similarity_metric=None, plot_info=False):
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
	"""
	MAX_SAMP_RATE_RATIO_DENOMINATOR = int(1e6)
	MIN_LENGTH_MULTIPLIER = 100

	if similarity_metric is None:
		similarity_metric = xcorr_norm

	# Handle resampling with integer approximations of the sampling rate ratio (This can likely be improved)
	samp_rate_ratio = samp_rate1 / samp_rate2
	sample_rate_ratio_approx = Fraction(samp_rate_ratio).limit_denominator(MAX_SAMP_RATE_RATIO_DENOMINATOR)

	scaler_multiplier = int(
		max(len(true_signal), len(adj_signal)) * MIN_LENGTH_MULTIPLIER / sample_rate_ratio_approx.denominator)
	samp_rate_1_approx = sample_rate_ratio_approx.numerator * scaler_multiplier
	samp_rate_2_approx = sample_rate_ratio_approx.denominator * scaler_multiplier

	resampled1 = scipy.signal.resample(true_signal, samp_rate_1_approx)
	resampled2 = scipy.signal.resample(adj_signal, samp_rate_2_approx)

	# print("Sample rate ratio approximation error:", samp_rate_ratio - float(sample_rate_ratio_approx))

	# Get the points of interest within the signals
	peaks1 = scipy.signal.find_peaks(resampled1, prominence=1)[0] ############ SHOULD LOOK INTO USING find_peaks_cwt #############
	peaks2 = scipy.signal.find_peaks(resampled2, prominence=1)[0]
	valleys1 = scipy.signal.find_peaks(-resampled1, prominence=1)[0]
	valleys2 = scipy.signal.find_peaks(-resampled2, prominence=1)[0]

	# Combine the peaks with the valleys for each signal
	poi1 = np.union1d(peaks1, valleys1)
	poi2 = np.union1d(peaks2, valleys2)

	if plot_info:
		#print("%d points of interest in signal 1\n%d points of intereste in signal 2"%(len(poi1), len(poi2)))
		time_step = (true_signal_times[-1] - true_signal_times[0])/(len(resampled1) - 1)

		plt.plot(resampled1)
		plt.plot(resampled2)
		plt.plot(poi1, resampled1[poi1], "x")
		plt.plot(poi2, resampled2[poi2], 'x')
		plt.xlabel("Time Steps (%.5e seconds)" % time_step)
		plt.title("Resampled Data with Marked POI ")
		plt.show()

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
	time_increment = (true_signal_times[-1] - true_signal_times[0]) / (len(true_signal) - 1) ####DOUBLE CHECK THE DENOMINATOR######

	best_slices = (None, None, None)
	best_score = np.finfo(np.float32).min
	for loc1 in poi1:
		for loc2 in poi2:
			if abs(loc1 - loc2) * time_increment <= max_start_offset:
				slices = [slice1, slice2, _] = get_aligned_slices(true_signal, adj_signal, true_signal_times, loc1, loc2)
				new_score = similarity_metric(slice1, slice2)

				if new_score > best_score:
					best_score = new_score
					best_slices = slices

	return best_slices


def align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync, true_time_steps, true_samp_rate,
				  max_start_offset=None, plot_info=False):
	"""
	TODO:
	 - Plot against time rather than index number

	:param true_signal: An ndarray for the signal data which is 'correct'
	:param adjustable_signal: An ndarray for the signal data which should be adjusted
	:param true_sync: The sync signal for the 'correct' signal
	:param adjustable_sync: The sync signal for the adjustable signal
	:param true_time_steps: The times associated with the true_signal data
	:param true_samp_rate: The sampling rate for the 'correct' signal
	:param max_start_offset: The maximum starting time difference to be checked allowed when trying to sync the signals
	:param plot_info: If the original and aligned signals should be plotted by matplotlib
	"""
	if plot_info:
		plt.plot(true_signal)
		plt.plot(adjustable_signal)
		plt.title("Original Signals")
		plt.xlabel("Time")
		plt.show()

	# If the maximum start offset is not given, set it to one fourth of the true signal's length
	if max_start_offset is None:
		max_start_offset = (true_time_steps[-1] - true_time_steps[0]) / 4

	time_increment = (true_time_steps[-1] - true_time_steps[0])/(len(true_time_steps)-1)

	# Calculate the adjustable signal's sampling rate
	sample_rate_ratio = get_sample_rate_ratio(true_sync, adjustable_sync, time_increment, plot_info)
	adjustable_samp_rate = true_samp_rate / sample_rate_ratio

	# Align the signals
	aligned = [truth_aligned, adjustable_aligned, aligned_time_steps] = resample_slide_and_compare(
		true_signal,
		adjustable_signal,
		true_time_steps,
		true_samp_rate,
		adjustable_samp_rate,
		max_start_offset=max_start_offset,
		plot_info=plot_info)

	if plot_info:
		plt.plot(truth_aligned)
		plt.plot(adjustable_aligned)
		plt.title("Aligned Signals")
		plt.xlabel("Time")
		plt.show()

	return aligned


def load_csv_data(base_dir):
	"""
	TODO:
	 - Have this take in file path+name for all the files, rather than the base directory

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
		"true_time": database['a_dev_accel']['time'],
		"adj_sync_time": database['s_dev_sync']['time'],
	}
