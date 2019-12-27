import numpy as np
from UTickSynchronization.time_sync import load_csv_data, get_sample_rate_ratio, align_signals

def test_sample_rate_ratio():
	for_sampling = np.sin(np.arange(1e6)/100)  # The division by a scaler is that integer sampling rates are reasonable

	testing_sample_rates = [1, 2, 3, 5, 8]
	for samp_rate_1 in testing_sample_rates:
		for samp_rate_2 in testing_sample_rates:
			signal_1 = for_sampling[::samp_rate_1]
			signal_2 = for_sampling[::samp_rate_2]

			max_len = min(len(signal_1), len(signal_2))
			signal_1 = signal_1[:max_len]
			signal_2 = signal_2[:max_len]

			true_ratio = samp_rate_1 / samp_rate_2

			ratio_error = get_sample_rate_ratio(signal_1, signal_2, true_timestep=1) - true_ratio
			assert abs(ratio_error) < .01


def test_align_signals():
	"""
	TODO:
	 - Have this not slice the differently sized signals.  Not sure why this is done
	 - Ensure terminology is correct (specifically sampling rate vs. frequency)
	"""
	TEST_DATA_DIR = "\\\\Mide2007\\Projects\\A6\\Design\\Software\\Sample_Data"

	test_data_dict = load_csv_data(TEST_DATA_DIR)

	whole_signal = test_data_dict['true_signal'] ############## HANDLE THE XYZ CHANNELS ###################
	whole_sync = test_data_dict['true_sync']
	whole_times = test_data_dict['true_time']

	true_sample_rate = (whole_times[-1] - whole_times[0])/(len(whole_times)-1)

	true_samp_rate = 3
	adjustable_samp_rate = 2
	true_offset = 0
	adjustable_offset = 100

	true_signal = whole_signal[true_offset:: true_samp_rate]
	adjustable_signal = whole_signal[adjustable_offset:: adjustable_samp_rate]

	true_sync = whole_sync[true_offset:: true_samp_rate]
	adjustable_sync = whole_sync[adjustable_offset:: adjustable_samp_rate]

	max_signal_len = min(len(true_signal), len(adjustable_signal))
	true_signal, adjustable_signal = true_signal[:max_signal_len], adjustable_signal[:max_signal_len]

	max_sync_len = min(len(true_sync), len(adjustable_sync))
	true_sync, adjustable_sync = true_sync[:max_sync_len], adjustable_sync[:max_sync_len]

	align_signals(true_signal, adjustable_signal, true_sync, adjustable_sync,
				  whole_times, true_sample_rate, plot_info=True)



def test_using_pete_data():
	TEST_DATA_DIR = "\\\\Mide2007\\Projects\\A6\\Design\\Software\\Sample_Data"

	test_data_dict = load_csv_data(TEST_DATA_DIR)

	whole_signal = test_data_dict['true_signal']  ############## HANDLE THE XYZ CHANNELS ###################
	whole_sync = test_data_dict['true_sync']
	whole_times = test_data_dict['true_time']

	adjust_signal = test_data_dict['adj_signal']
	adjust_sync = test_data_dict['adj_sync']

	TRUE_SAMPLE_RATE = (whole_times[-1] - whole_times[0]) / (len(whole_times) - 1)
	aligned = align_signals(whole_signal, adjust_signal, whole_sync, adjust_sync,
							whole_times, TRUE_SAMPLE_RATE, plot_info=True)


if __name__ == '__main__':
	test_sample_rate_ratio()

	test_align_signals()

	test_using_pete_data()

