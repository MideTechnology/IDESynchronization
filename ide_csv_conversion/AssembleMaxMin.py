"""
@author: Peter Scheidler, Mide Technology
"""
import pandas as pd
import numpy as np
import os
from CsvAvgStd import decode_summary_filename, listdir_full_path

WorkDir = """E:\\NavyThermo_Pete\\CSVs\\Results"""
OutputFile = """OverallSummary.csv"""

def get_file_data(file):
    file_info = decode_summary_filename(file)
    if file_info is None:
        return
    file_data = pd.read_csv(file)
    for col in file_data.keys():
        file_info['%s %s %s Max' % (file_info['chan'], col, file_info['type'])] = np.max(file_data[col])
        file_info['%s %s %s Min' % (file_info['chan'], col, file_info['type'])] = np.min(file_data[col])
    del file_info['type']
    del file_info['chan']
    return file_info

if __name__ == "__main__":
    file_list = listdir_full_path(WorkDir)
    file_list.sort()
    main_data = {}
    col_list = []
    for filename in file_list:
        processed_data = get_file_data(filename)
        if processed_data is None:
            continue
        dev = processed_data['device']
        del processed_data['device']
        if dev not in main_data:
            main_data[dev] = {}
        for col, data in processed_data.iteritems():
            main_data[dev][col] = data
            if col not in col_list:
                col_list.append(col)
    with open(os.path.join(WorkDir, OutputFile), 'w+') as out_file:
        out_file.write('Device,')
        out_file.write(','.join(['%s'%(x) for x in col_list])+'\n')
        for dev in main_data.keys():
            out_file.write(dev+',')
            out_file.write(','.join(['%s' % (main_data[dev].get(col, 'XXX')) for col in col_list]))
            out_file.write('\n')
    print("Done!")