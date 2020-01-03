from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from UTickSynchronization.time_sync import load_csv_data, get_sample_rate_ratio, align_signals, sync_and_create_new_ide


def test_sample_rate_ratio():
	for_sampling = np.sin(np.arange(1e6)/100)  # The division by a scaler is that integer sampling rates are reasonable

	testing_sample_rates = [1, 2, 5, 8]
	for samp_rate_1 in testing_sample_rates:
		for samp_rate_2 in testing_sample_rates:
			signal_1 = for_sampling[::samp_rate_1]
			signal_2 = for_sampling[50::samp_rate_2]

			true_ratio = samp_rate_1 / samp_rate_2
			ratio_error = get_sample_rate_ratio(signal_1, signal_2, true_timestep=1, adjust_timestep=1) - true_ratio

			assert abs(ratio_error) < .01


def test_align_signals():
	"""
	A test that samples from a single signal and it's corresponding sync signal to produce two different new signals
	and syncs of different sampling rates.  It then tests the code's ability to sync the signals back together.

	TODO:
	 - Ensure terminology is correct (specifically sampling rate vs. frequency)
	"""
	TEST_DATA_DIR = "\\\\Mide2007\\Projects\\A6\\Design\\Software\\Sample_Data"

	test_data_dict = load_csv_data(TEST_DATA_DIR)

	whole_signal = test_data_dict['true_signal']
	whole_sync = test_data_dict['true_sync']
	whole_times = test_data_dict['true_time']

	true_sample_rate = (whole_times[-1] - whole_times[0])/(len(whole_times)-1)

	true_samp_rate = 2
	adjustable_samp_rate = 3
	true_offset = 1000
	adjustable_offset = 0

	true_signal = whole_signal[true_offset:: true_samp_rate]
	adjustable_signal = whole_signal[adjustable_offset:: adjustable_samp_rate]
	true_times = whole_times[true_offset:: true_samp_rate]

	true_sync = whole_sync[true_offset:: true_samp_rate]
	adjustable_sync = whole_sync[adjustable_offset:: adjustable_samp_rate]

	align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync,
				  true_times[:len(true_signal)], true_times[:len(adjustable_signal)], true_sample_rate, plot_info=True)


def test_using_pete_data():
	"""
	Testing the ability of the code to sync example data.
	"""
	TEST_DATA_DIR = "\\\\Mide2007\\Projects\\A6\\Design\\Software\\Sample_Data"

	test_data_dict = load_csv_data(TEST_DATA_DIR)

	true_signal = test_data_dict['true_signal']
	true_sync = test_data_dict['true_sync']
	true_times = test_data_dict['true_time']

	adjust_signal = test_data_dict['adj_signal']
	adjust_sync = test_data_dict['adj_sync']
	adjust_times = test_data_dict['adj_time']

	TRUE_SAMPLE_RATE = (true_times[-1] - true_times[0]) / (len(true_times) - 1)
	align_signals(true_signal, adjust_signal, true_sync, adjust_sync, true_times, adjust_times, TRUE_SAMPLE_RATE,
				  plot_info=True)


def test_synchronization_from_ide_to_aligned_csv():
	ide_path = "\\\\Mide2007\\Projects\\A6\\Design\\Software\\Sample_Data\\ide_files"
	true_ide = "ANA00008_T2.IDE"
	adj_ide = "SSS00001_T2.IDE"

	sync_and_create_new_ide(ide_path, true_ide, adj_ide)

	to_plot_name = ["ANA00008_T2_Ch80.csv", "SSS00001_T2_Ch80.csv", "SSS00001_T2_Ch80_adjusted.csv"]
	to_plot = list(map(lambda x: "%s\\%s"%(ide_path, x), to_plot_name))
	for j, fn in enumerate(to_plot):
		npa = np.genfromtxt(fn, delimiter=',', skip_header=1)
		plt.plot(npa[:, 0], npa[:, -1], label=to_plot_name[j])

	plt.legend()
	plt.show()



if __name__ == '__main__':
	test_sample_rate_ratio()

	test_align_signals()

	test_using_pete_data()

	test_synchronization_from_ide_to_aligned_csv()
