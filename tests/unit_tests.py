from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys
import wx
import numpy as np
sys.path.insert(0,'..')
from IDESynchronization.time_sync import load_csv_data, get_sample_rate_ratio, align_signals, sync_and_create_new_csv
from IDESynchronization.sync_ui import SynchronizationWindow
import matplotlib.pyplot as plt

def make_sine(frequency, sample_period, length, offset_points=0):
    offset_time = offset_points*sample_period
    time = np.arange(offset_points*sample_period, length*sample_period+offset_time, sample_period)
    sin = np.sin(2*np.pi*time*frequency)
    return sin


def test_sample_rate_ratio():
    passed = True

    true_sample_periods = [0.0002, 0.0001, 0.00005, 0.00015]
    adj_target_sample_periods = [0.00005, 0.0002, 0.0001, 0.00015]
    # multiplier is 1 + error
    adj_sample_period_multipliers = [1+0.01, 1-0.01, 1+0, 1+20/1e6]
    sync_frequency = 800
    sample_length = 100000
    adj_offset = 97
    for true_sample_period in true_sample_periods:
        for adj_target_sample_period in adj_target_sample_periods:
            for adj_sample_period_multiplier in adj_sample_period_multipliers:
                adj_actual_sample_period = adj_target_sample_period * adj_sample_period_multiplier
                true_sync_signal = make_sine(sync_frequency, true_sample_period, sample_length)
                adj_sync_signal = make_sine(sync_frequency, adj_actual_sample_period,
                                            sample_length, offset_points=adj_offset)
                true_ratio = adj_target_sample_period / adj_actual_sample_period
                calculated_ratio = get_sample_rate_ratio(true_sync_signal, adj_sync_signal, true_timestep=true_sample_period,
                                                    adjust_timestep=adj_target_sample_period)
                ratio_error = calculated_ratio - true_ratio

                if abs(ratio_error) >= .01:
                    passed = False
                assert abs(ratio_error) < .01
    return passed

def test_align_signals():
    """
    A test that samples from a single signal and it's corresponding sync signal to produce two different new signals
    and syncs of different sampling rates.  It then tests the code's ability to sync the signals back together.

    TODO:
     - Ensure terminology is correct (specifically sampling rate vs. frequency)
    """
    TEST_DATA_DIR = "tests\\data"

    test_data_dict = load_csv_data(TEST_DATA_DIR)

    whole_signal = test_data_dict['true_signal']
    whole_sync = test_data_dict['true_sync']
    whole_times = test_data_dict['true_signal_time']

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

    align_signals(true_signal,
                  adjustable_signal,
                  true_sync,
                  adjustable_sync,
                  true_times[:len(true_signal)],
                  true_times[:len(adjustable_signal)],
                  true_sample_rate,
                  lambda x: None,
                  plot_info=True)


def test_using_pete_data():
    """
    Testing the ability of the code to sync example data.
    """
    TEST_DATA_DIR = "tests\\data"

    test_data_dict = load_csv_data(TEST_DATA_DIR)

    true_signal = test_data_dict['true_signal']
    true_sync = test_data_dict['true_sync']
    true_times = test_data_dict['true_time']

    adjust_signal = test_data_dict['adj_signal']
    adjust_sync = test_data_dict['adj_sync']
    adjust_times = test_data_dict['adj_time']

    TRUE_SAMPLE_RATE = (true_times[-1] - true_times[0]) / (len(true_times) - 1)
    align_signals(true_signal,
                  adjust_signal,
                  true_sync,
                  adjust_sync,
                  true_times,
                  adjust_times,
                  TRUE_SAMPLE_RATE,
                  lambda x: None,
                  plot_info=True)

def test_synchronization_from_ide_to_aligned_csv():
    true_ide = "tests\\data\\ANA00008_T2.IDE"
    adj_ide = "tests\\data\\SSS00001_T2.IDE"
    output_path = "tests\\data"

    sync_and_create_new_csv(
        true_ide,
        adj_ide,
        output_path,
        show_signal_plots=True)

    # Load and plot the data for the original signals and the adjusted signals from the csv files created
    to_plot_name = ["ANA00008_T2_Ch80.csv", "SSS00001_T2_Ch80.csv", "SSS00001_T2_Ch80_adjusted.csv"]
    to_plot = list(map(lambda x: "%s\\%s"%(output_path, x), to_plot_name))
    for j, fn in enumerate(to_plot):
        npa = np.genfromtxt(fn, delimiter=',', skip_header=1)
        plt.plot(npa[:, 0], npa[:, -1], label=to_plot_name[j])

    plt.legend()
    plt.show(block=False)

def test_synchronization_through_ui():
    """
    NOTES:
     - The UI window may look a bit funky, but that's okay
    """
    app = wx.App(False)
    frame = SynchronizationWindow(None, "UTick Signal Synchronization")

    true_ide = "tests\\data\\ANA00008_T2.IDE"
    adj_ide = "tests\\data\\SSS00001_T2.IDE"
    output_path = "tests\\data"

    # Set the paths in the ui elements
    frame.true_ide_edit_field.write(true_ide)
    frame.adj_ide_edit_field.write(adj_ide)
    frame.output_edit_field.write(output_path)

    frame.graph_check_box.SetValue(True)

    frame.synchronize()

    # Load and plot the data for the original signals and the adjusted signals from the csv files created
    to_plot_name = ["ANA00008_T2_Ch80.csv", "SSS00001_T2_Ch80.csv", "SSS00001_T2_Ch80_adjusted.csv"]
    to_plot = list(map(lambda x: "%s\\%s"%(output_path, x), to_plot_name))
    for j, fn in enumerate(to_plot):
        npa = np.genfromtxt(fn, delimiter=',', skip_header=1)
        plt.plot(npa[:, 0], npa[:, -1], label=to_plot_name[j])

    plt.legend()

    frame.Close()
    plt.show()


if __name__ == '__main__':
    if test_sample_rate_ratio():
        print("Sample Rate Ratio test passed")

    # test_align_signals()

	# test_using_pete_data()

    # test_synchronization_from_ide_to_aligned_csv()

    test_synchronization_through_ui()
