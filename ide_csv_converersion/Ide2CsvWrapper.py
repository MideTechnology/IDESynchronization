"""
@author: Peter Scheidler, Mide Technology
"""
import UTickSynchronization.ide_csv_converersion.ide_helpers as ide_helpers
import subprocess
import os

class Ide2CsvWrapper(object):
    """ This uses the ide2csv executable to convert IDE files into CSVs
    """
    ## TODO: Verify all these are the right values!

    def __init__(self, ide_files, converter="ide2csv_64b.exe", output_type="csv",
                 channels=ide_helpers.channels_by_name.keys(), name=True, utc=True,
                 remove_dc=False, output_path=None):
        ## TODO: Check existance of converter, throw an error
        if isinstance(ide_files, str):
            self.ide_files = [ide_files]
        else:
            self.ide_files = ide_files
        self.converter = converter
        self.output_type = output_type
        self.channels = channels
        self.name = name
        self.utc = utc
        self.remove_dc = remove_dc
        self.output_path = output_path


    def run(self):
        call_list = [self.converter]
        if self.output_path:
            if not os.path.isdir(self.output_path):
                os.makedirs(self.output_path)
            call_list.append("-o%s" % (self.output_path))
        if self.output_type:
            call_list.append("-t%s" % (self.output_type))
        channel_ids = ide_helpers.channel_desc_to_id(self.channels)
        for channel_id in channel_ids:
            call_list.append("-c%s" % (channel_id))
        if self.remove_dc:
            call_list.append("-m1")
        else:
            call_list.append("-m0")
        if self.utc:
            call_list.append("-u")
        if self.name:
            call_list.append("-n")
        input_files = []
        for item in self.ide_files:
            if os.path.isdir(item):     # If we are given a directory, grab all IDEs in that dir
                input_files.append(os.path.join(item, "*.ide"))
            elif os.path.exists(item):
                input_files.append(item)
        call_list += input_files
        print(call_list)
        subprocess.call(call_list)
