import os
import pandas as pd
import numpy as np
import time

OutputBaseName = "Results"
InputDir  = "E:\\NavyThermo_Pete\\Csvs"
OutputDir = os.path.join(InputDir, "Results")
#Devices = ["31", "32", "35", "37", "42", "43", "44", "24"]
#Devices = ["14", "17", "20", "21", "26", "27", "29", "30"]
Devices = ["16"]


def get_summary_filenames(device, chan, output_dir):
    stddev_filename = sanitize_csv_name('%s_%s_StdDev' % (device, chan), dir=output_dir)
    mean_filename   = sanitize_csv_name('%s_%s_Mean' % (device, chan), dir=output_dir)
    return stddev_filename, mean_filename

def decode_summary_filename(filename_in):
    filename = os.path.basename(filename_in)
    if not filename.endswith('.csv'):
        print("Problem! Got %s, not a CSV as a summary filename" % (filename_in))
        return None
    parts = filename.strip(".csv").split("_")
    if len(parts) != 3:
        print ("Problem! Not enough parts in %s (%s)" % (filename_in, filename))
        return None
    return {'device':parts[0], 'chan':parts[1], 'type':parts[2]}

def sanitize_csv_name(name, dir=None):
    if not isinstance(name, str):
        print("Raise an error, name is not a string")
        return name
    temp_name = name
    if not temp_name.endswith(".csv"):
        temp_name += ".csv"
    if dir is not None:
        temp_name = os.path.join(dir, temp_name)
    if os.path.isfile(temp_name):
        return temp_name
    print("NOTE: file %s does not exit" % temp_name)
    return temp_name

def listdir_full_path(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]

def get_subdirs(base_dir):
    input_list = os.listdir(base_dir)
    dirs = [d for d in listdir_full_path if os.path.isdir(d)]
#    dirs.sort()
    return dirs

# get average and standard deviation of a single file
# return format: {'Mean '+column_name : Value, 'StdDev '+column_name : Value}, [column_names]
def general_file_process(input_name):
    # Read in the CSV
    file_name = sanitize_csv_name(input_name)
    chan_data = pd.read_csv(file_name)
    # Clean up the column names
    original_cols = chan_data.columns
    new_cols = [c.replace('"','').strip() for c in original_cols]
    rename_mapper = dict(zip(original_cols, new_cols))
    chan_data = chan_data.rename(index=str, columns=rename_mapper)
    # Average out the results
    processed_data = {}
    for col in chan_data.keys():
        processed_data['Mean '+col] = np.mean(chan_data[col])
        processed_data['StdDev '+col] = np.std(chan_data[col])
    del chan_data   # free up the memory
    processed_data['File'] = file_name
    return processed_data, new_cols

def dir_process_np(input_dir, device='D', output_dir=OutputDir, max_buffer_size=2600):
    stddev_filename = sanitize_csv_name(device+'_StdDev', dir=output_dir)
    mean_filename   = sanitize_csv_name(device+'_Mean', dir=output_dir)
    input_filenames = [f for f in listdir_full_path(input_dir) if os.path.isfile(d) and d.lower().endswith('.csv')]
    input_filenames.sort()
    data_holder = pd
    for i, filename in enumerate(input_filenames):
        pass
    ## You should now have the CSV file created, name = ide_file+"_Ch08.csv"
    csv_file = """C:/Users/pscheidler/Documents/Slam/Navy Thermo/FinalData/D15/20180827/SSX41947_Ch08.csv"""
    reader = pd.read_csv(csv_file)
    
    out_file = """C:/Users/pscheidler/Documents/Slam/Navy Thermo/FinalData/D15/20180827/test.csv"""
    rename_mapper = {' "A"' : 'A',' "B"' : 'B',' "C"' : 'C',' "D"' : 'D'}
    input_file = "name"
    input_file = sanitize_csv_name(input_file)
    

def dir_process_text(input_dir, device='D', chan="", output_dir=OutputDir, max_buffer_size=2600, response_counter=1000):
    first_read = True
    stddev_filename = sanitize_csv_name('%s_%s_StdDev' % (device, chan), dir=output_dir)
    with open(stddev_filename, 'w+') as f:
        f.write('Time,StdDev A,StdDev B,StdDev C,StdDev D,StdDev Time,File\n')
    mean_filename   = sanitize_csv_name('%s_%s_Mean' % (device, chan), dir=output_dir)
    with open(mean_filename, 'w+') as f:
        f.write('Time,Mean A,Mean B,Mean C,Mean D,File\n')
    print stddev_filename, mean_filename
    input_filenames = [f for f in listdir_full_path(input_dir) if (os.path.isfile(f) and f.lower().endswith('.csv') and chan.lower() in os.path.basename(f).lower())]
    input_filenames.sort()
    stddev_buffer = ""
    mean_buffer = ""
    num_files = len(input_filenames)
    start = time.time()
    for i, filename in enumerate(input_filenames):
        if i % response_counter == response_counter-1:
            elapsed_time = time.time() - start
            time_left = elapsed_time * num_files / i
            end_time_str = time.strftime("%H:%M", time.localtime(start + time_left))
            print "%d/%d, %d seconds left, done at %s" % (i+1, num_files, time_left, end_time_str)
        # If the buffers are full, write them and empty out
        if len(stddev_buffer) > max_buffer_size:
            with file(stddev_filename, 'a') as f:
                f.write(stddev_buffer)
            stddev_buffer = ""
        if len(mean_buffer) > max_buffer_size:
            with file(mean_filename, 'a') as f:
                f.write(mean_buffer)
            mean_buffer = ""
        # Get the processed data, add it to the buffer
        file_data, file_columns = general_file_process(filename)
        if first_read:
            first_read = False
            mean_cols = ['Mean %s' % c for c in file_columns] + ['File']
            stddev_cols = ['StdDev %s' % c for c in file_columns]
            stddev_cols[0] = 'Mean Time'
            stddev_cols.append('StdDev Time')
            stddev_cols.append('File')
            mean_buffer = ",".join(mean_cols) + '\n'
            stddev_buffer = ",".join(stddev_cols) + '\n'
        stddev_buffer += ','.join([str(file_data[c]) for c in stddev_cols]) + '\n'
        mean_buffer   += ','.join([str(file_data[c]) for c in mean_cols]) + '\n'
    # Write out any remaining data
    with file(stddev_filename, 'a') as f:
        f.write(stddev_buffer)
    with file(mean_filename, 'a') as f:
        f.write(mean_buffer)
        
if __name__ == "__main__":
    for device in Devices:
        start_time = time.time()
        dir_process_text(os.path.join(InputDir, device), device=device, max_buffer_size=10000, chan='Thermo')
        dir_process_text(os.path.join(InputDir, device), device=device, max_buffer_size=10000, chan='TPH')
        dir_process_text(os.path.join(InputDir, device), device=device, max_buffer_size=10000, chan='Accel')
        
        print("** Elapsed Time = %d" % (time.time() - start_time))
    