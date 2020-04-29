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
    whole_sync_times = test_data_dict['true_sync_time']

    true_samp_rate = 3
    adjustable_samp_rate = 2
    true_offset = 0
    adjustable_offset = 200

    for true_offset in [0,100, 200]:
        for adjustable_offset in [0, 200]:   #THIS GROSS LOOP IS ONLY HERE BECAUSE I'M TESTING SOMETHING########.
    # for true_offset in [0,200]:
    #     for adjustable_offset in [0, 200]:   #THIS GROSS LOOP IS ONLY HERE BECAUSE I'M TESTING SOMETHING########
    # for (true_offset, adjustable_offset) in [(0, 200), (200, 0)]:
    # for (true_offset, adjustable_offset) in [(0, 0), (200, 200)]:
            true_signal = whole_signal[true_offset:: true_samp_rate]
            adjustable_signal = whole_signal[adjustable_offset:: adjustable_samp_rate]

            true_times = whole_times[true_offset:: true_samp_rate]
            adj_times = whole_times[adjustable_offset:: adjustable_samp_rate]

            print("TIMES", true_times[:5], adj_times[:5])

            true_sync = whole_sync[true_offset:: true_samp_rate]
            adjustable_sync = whole_sync[adjustable_offset:: adjustable_samp_rate]

            true_sync_times = whole_sync_times[true_offset:: true_samp_rate]
            adj_sync_times = whole_sync_times[adjustable_offset:: adjustable_samp_rate]

            # x[0] + np.cumsum(np.diff((x - x[0]), prepend=[0]))
            adj_sync_times = adj_sync_times[0] + np.cumsum(np.diff((adj_sync_times - adj_sync_times[0]), prepend=[0]) * (true_samp_rate / adjustable_samp_rate))
            adj_times = adj_times[0] + np.cumsum(np.diff((adj_times - adj_times[0]), prepend=[0]) * (true_samp_rate / adjustable_samp_rate))
            # adj_sync_times *= true_samp_rate / adjustable_samp_rate

            # plt.plot(true_times, true_signal)
            # plt.plot(adj_times, adjustable_signal)
            # plt.title("%d, %d"%(true_samp_rate, adjustable_samp_rate))
            # plt.show()

            # stuff = [adjustable_signal,true_signal, adjustable_sync,true_sync,true_times[:len(adjustable_signal)],true_times[:len(true_signal)],true_sync_times[:len(adjustable_sync)],true_sync_times[:len(true_sync)]]

            # print("LENGTHS", list(map(len, stuff)))
            if true_times[0] < adj_times[0]:

                align_signals(adjustable_signal,true_signal,
                              adjustable_sync,true_sync,
                              # true_times[:len(adjustable_signal)],true_times[:len(true_signal)],
                              adj_times, true_times,
                              # true_sync_times[:len(adjustable_sync)],true_sync_times[:len(true_sync)],
                              adj_sync_times, true_sync_times,
                              lambda x: None,
                              plot_info=True)

                plt.suptitle("Offsets = (%d, %d), Signals Switched" % (true_offset, adjustable_offset), fontweight='bold')
                plt.show()

            else:

                align_signals(true_signal,
                              adjustable_signal,
                              true_sync,
                              adjustable_sync,
                              true_times,
                              adj_times,
                              # true_times[:len(true_signal)],
                              # true_times[:len(adjustable_signal)],
                              # true_sync_times[:len(true_sync)],
                              # true_sync_times[:len(adjustable_sync)],
                              true_sync_times,
                              adj_sync_times,
                              lambda x: None,
                              plot_info=True)
                plt.suptitle("Offsets = (%d, %d)" % (true_offset, adjustable_offset), fontweight='bold')
                plt.show()


def test_using_pete_data(testing_data_dir="tests\\data"):
    """
    Testing the ability of the code to sync example data.

    TODO:
     - Rename this function
     - Give this function a better description
    """
    data = load_csv_data(testing_data_dir)

    align_signals(data['adj_signal'], data['true_signal'],
                  data['adj_sync'], data['true_sync'],
                  data['adj_signal_time'], data['true_signal_time'],
                  data['adj_sync_time'], data['true_sync_time'],
                  lambda x: None,
                  plot_info=True)
    plt.show()

    align_signals(data['true_signal'], data['adj_signal'],
                  data['true_sync'], data['adj_sync'],
                  data['true_signal_time'], data['adj_signal_time'],
                  data['true_sync_time'], data['adj_sync_time'],
                  lambda x: None,
                  plot_info=True)
    plt.show()




def test_synchronization_from_ide_to_aligned_csv():
    true_ide = "tests\\data\\ANA00008_T2.IDE"
    adj_ide = "tests\\data\\SSS00001_T2.IDE"
    output_path = "tests\\data"

    sync_and_create_new_csv(
        true_ide,
        adj_ide,
        output_path,
        show_signal_plots=True)

    # # Load and plot the data for the original signals and the adjusted signals from the csv files created
    # to_plot_name = ["ANA00008_T2_Ch80.csv", "SSS00001_T2_Ch80.csv", "SSS00001_T2_Ch80_adjusted.csv"]
    # to_plot = list(map(lambda x: "%s\\%s"%(output_path, x), to_plot_name))
    # for j, fn in enumerate(to_plot):
    #     npa = np.genfromtxt(fn, delimiter=',', skip_header=1)
    #     plt.plot(npa[:, 0], npa[:, -1], label=to_plot_name[j])
    #
    # plt.legend()
    plt.show()#block=False)  ###### SHOULD GO BACK TO BLOCKING WHEN DONE TESTING SO MUCH ###################

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
    # to_plot_name = ["ANA00008_T2_Ch80.csv", "SSS00001_T2_Ch80.csv", "SSS00001_T2_Ch80_adjusted.csv"]
    # to_plot = list(map(lambda x: "%s\\%s"%(output_path, x), to_plot_name))
    # for j, fn in enumerate(to_plot):
    #     npa = np.genfromtxt(fn, delimiter=',', skip_header=1)
    #     plt.plot(npa[:, 0], npa[:, -1], label=to_plot_name[j])
    #
    # plt.legend()

    frame.Close()
    plt.show()

    del app  # Pretty sure this is better practice than just leaving it (though it may destroy itself, should look into)


if __name__ == '__main__':
    if test_sample_rate_ratio():
        print("Sample Rate Ratio test passed")

    # Was being used to help locate the asymmetry bug, will put back to normal when done!
    # test_align_signals()

    test_using_pete_data()

    test_synchronization_from_ide_to_aligned_csv()

    test_synchronization_through_ui()
