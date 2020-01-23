import Ide2CsvWrapper
import ide_helpers
import os
import time

## We will append the dir name to the base, grab any files in that dir, and search 1 dir deep
DeviceBaseDir = "E:\\NavyThermo_Pete\\20190118_LCS_16"
DeviceDirNames = ["16"]
#DeviceDirNames = ["14", "17", "20", "21", "26", "27", "29", "30"]
#DeviceDirNames = ["31", "32", "35", "37", "42", "43", "44", "24"]
OutputDir = "E:\\NavyThermo_Pete\\Csvs"
SeparateDeviceDirs = False
#ChannelsToConvert = ['Main Accel', 'TPH', 'DC Accel']
#ChannelsToConvert = ['TPH', 'DC Accel']
ChannelsToConvert = ['Main Accel']
ChannelIds = {8 : 'Thermo', 59: 'TPH', 32:'Accel' }
StartTime = 0
# ide2csv_64b -oC:\Users\pscheidler\Documents\Slam\Navy_Thermo\FinalData\Csvs\D15 -tcsv -c8 -m0 -u -n C:\Users\pscheidler\Documents\Slam\Navy_Thermo\FinalData\D15\20180827\SSX42093.IDE

def get_log_name(dev):
    LogDir = "E:\\NavyThermo_Pete\\20190118_LCS_16"
    ErrorLog = "BatchLog.txt"
    return os.path.join(LogDir, "%s_%s" % (dev, ErrorLog))

def start_error_log(dev):
    with file(get_log_name(dev), 'w') as f:
        f.write("IDE File,IDE Size,CSV Number,Time,Errors\n")

def update_log(ide_filename, dev, number, error_buffer, error_log_buffer, log_buffer_size=10000):
    file_size = os.stat(ide_filename).st_size
    error_log_buffer += "%s,%d,%d,%d,%s\n" % (ide_filename, file_size, number, time.time(), error_buffer)
    if len(error_log_buffer) > log_buffer_size:
        send_to_log(dev, error_log_buffer)
        error_log_buffer = ""
    return error_log_buffer

def send_to_log(dev, error_log_buffer):
    with file(get_log_name(dev), 'a') as f:
        f.write(error_log_buffer)

def log_error(error, buffer):
    return buffer + error.strip() + " "

def get_ide_files(dir):
    input_list = os.listdir(dir)
    files = [os.path.join(dir, f) for f in input_list if os.path.isfile(os.path.join(dir, f)) and f.lower().endswith('.ide') ]
    files.sort()
    return files

def get_subdirs(base_dir):
    input_list = os.listdir(base_dir)
    dirs = [os.path.join(base_dir, d) for d in input_list if os.path.isdir(os.path.join(base_dir, d))]
    dirs.sort()
    return dirs

def get_csv_filename(ide_name, new_base_dir=None, channel=8):
    csv_name = "%s_Ch%02d.csv" % (ide_name[:-4], channel)
    if new_base_dir:
        csv_name = os.path.join(new_base_dir, os.path.basename(csv_name))
    return csv_name
    
def get_indexed_filename(device_str, type, index_num, base_dir):
    return os.path.join(base_dir, "%s_%s_%05d.csv" % (device_str, type, index_num))

def get_time_filename(device_str, type, base_dir, date, ide_file):
    return os.path.join(base_dir, "%s_%s_%s_%s.csv" % (device_str, type, date, ide_file[-9:-4]))

def convert_dir(input_dir, output_dir, start_index=-1, device_str='D', convert_size=25, channel_ids=ChannelIds):
    file_list = get_ide_files(input_dir)
    if len(file_list) < 1:
        return start_index
    top_dir = file_list[0].split('\\')[-2]
    if convert_size < 1:
        convert_size = len(file_list)
    for idx in xrange(0, len(file_list), convert_size):
        converter = Ide2CsvWrapper.Ide2CsvWrapper(file_list[idx:idx+convert_size], channels=ChannelsToConvert, output_path = output_dir)
        converter.run()
    if start_index == -1:
        return 0
    error_log_buffer = ""
    for ide_file in file_list:
        error_buffer = ""
        for chan, name in channel_ids.iteritems():
            old_name = get_csv_filename(ide_file, new_base_dir=output_dir, channel=chan)
            # new_name = get_indexed_filename(device_str, name, start_index, output_dir)
            new_name = get_time_filename(device_str, name, output_dir, top_dir, ide_file)
            if not os.path.isfile(old_name):
                error_buffer = log_error("Missing chan %d" % (chan), error_buffer)
            else:
#                print old_name, new_name
                os.rename(old_name, new_name)
        error_log_buffer = update_log(ide_file, device_str, start_index, error_buffer, error_log_buffer)
        start_index += 1
    send_to_log(device_str, error_log_buffer)
    return start_index


if __name__ == "__main__":
    for device_dir_name in DeviceDirNames:
        StartTime = time.time()
        start_error_log(device_dir_name)
        start_path = os.path.join(DeviceBaseDir, device_dir_name)
        output_path = os.path.join(OutputDir, device_dir_name)
        start_index = 0
        start_index = convert_dir(start_path, output_path, start_index=start_index, device_str=device_dir_name)
        subdirs = get_subdirs(start_path)
        for subdir in subdirs:
            start_index = convert_dir(subdir, output_path, start_index=start_index, device_str=device_dir_name)
        

