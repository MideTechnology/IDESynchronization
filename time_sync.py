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
	return np.sum(x*y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


def get_sample_rate_ratio(true_sync, adjust_sync, true_timestep):
	"""
	Get the ratio of the two sampling rates for the sync signals (after calculating the adjusted sync signal's
	sampling rate)
	"""
	true_fft = np.abs(np.fft.rfft(true_sync))
	true_fft_freqs = np.fft.rfftfreq(len(true_fft), d=true_timestep)

	true_sync_freq = true_fft_freqs[np.argmax(true_fft)]  # USE THIS TO VALIDATE THIS APPROACH AGAINST KNOWN VALUES

	adjust_fft = np.abs(np.fft.rfft(adjust_sync))

	adjust_sync_freq = true_fft_freqs[np.argmax(adjust_fft)]

	return true_sync_freq / adjust_sync_freq


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


def resample_slide_and_compare(data1, data2, data1_times, samp_rate1, samp_rate2, num_points_of_interest=500,
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

	# Handle resampling with integer approximations of the sampling rate ratio (This can likely be improved)
	samp_rate_ratio = samp_rate1 / samp_rate2
	sample_rate_ratio_approx = Fraction(samp_rate_ratio).limit_denominator(1e6)
	samp_rate_1_approx = sample_rate_ratio_approx.numerator
	samp_rate_2_approx = sample_rate_ratio_approx.denominator

	resampled1 = scipy.signal.resample(data1, samp_rate_1_approx)
	resampled2 = scipy.signal.resample(data2, samp_rate_2_approx)

	print("Sample rate ratio approximation error:", samp_rate_ratio - float(sample_rate_ratio_approx))

	# Get the points of interest within the signals
	max_locations1 = np.argpartition(resampled1, -num_points_of_interest)[-num_points_of_interest:]
	max_locations2 = np.argpartition(resampled2, -num_points_of_interest)[-num_points_of_interest:]

	# Check how similar the two signals are when aligning based on two points of interest (one in each signal)
	best_slices = []
	best_score = float('-inf')
	for loc1 in max_locations1:
		for loc2 in max_locations2:
			slices = [slice1, slice2, _] = get_aligned_slices(resampled1, resampled2, data1_times, loc1, loc2)
			new_score = similarity_metric(slice1, slice2)

			if new_score > best_score:
				best_score = new_score
				best_slices = slices

	return best_slices


def align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync, true_time_steps, true_samp_rate,
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
	time_increment = true_time_steps[1] - true_time_steps[0]

	# Calculate the adjustable signal's sampling rate
	sample_rate_ratio = get_sample_rate_ratio(true_sync, adjustable_sync, time_increment)
	adjustable_samp_rate = true_samp_rate / sample_rate_ratio

	# Align the signals
	aligned = [truth_aligned, adjustable_aligned, aligned_time_steps] = resample_slide_and_compare(
		true_signal,
		adjustable_signal,
		true_time_steps,
		true_samp_rate,
		adjustable_samp_rate)

	if plot_before_after:
		plt.plot(true_signal, true_time_steps)
		plt.plot(adjustable_signal, true_time_steps)
		plt.title("Original Signals")
		plt.xlabel("Time")
		plt.ylabel("Magnitude")
		plt.show()

		plt.plot(truth_aligned, aligned_time_steps)
		plt.plot(adjustable_aligned, aligned_time_steps)
		plt.title("Aligned Signals")
		plt.xlabel("Time")
		plt.ylabel("Magnitude")
		plt.show()

	return aligned
