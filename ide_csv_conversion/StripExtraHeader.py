"""
@author: Peter Scheidler, Mide Technology
"""
import os
import shutil
from CsvAvgStd import listdir_full_path

WorkDir = """E:\\NavyThermo_Pete\\CSVs\\Results"""
BackupDir = os.path.join(WorkDir, 'Backup')

if __name__ == "__main__":
    if not os.path.exists(BackupDir):
        os.mkdir(BackupDir)
    file_list = [f for f in listdir_full_path(WorkDir) if f.endswith('.csv')]
    for filename in file_list:
        print filename
        with open(filename, 'r') as f:
            _ = f.readline()
            start = f.tell()
            second = f.readline()
            if 'time' not in second.lower():
                continue
            f.seek(start)
            with open('tmp.csv', 'w+') as w:
                shutil.copyfileobj(f, w)
        backupname = os.path.join(BackupDir, os.path.basename(filename))
        os.rename(filename, backupname)
        os.rename('tmp.csv', filename)
    print("Done!")