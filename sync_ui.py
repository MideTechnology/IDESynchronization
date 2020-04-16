from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import wx
import os
import sys
sys.path.insert(0,'..')
from IDESynchronization.time_sync import sync_and_create_new_csv


class ProgressDialogWrapper:
    def __init__(self, progress_dialog):
        self.cur_num = 0
        self.progress_dialog = progress_dialog

    def __call__(self, text):
        self.progress_dialog.Update(self.cur_num, text)
        self.cur_num += 1
        wx.Yield()


class SynchronizationWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(500, 303))

        SPACER_SIZE = 10

        true_ide_file_label = wx.StaticText(self, wx.ID_ANY, "True Signal IDE File:")
        adj_ide_file_label = wx.StaticText(self, wx.ID_ANY, "Adjustable Signal IDE File:")
        output_dir_path_label = wx.StaticText(self, wx.ID_ANY, "Output Directory:")

        true_ide_browse_button = wx.Button(self, wx.ID_ANY, "Browse")
        adj_ide_browse_button = wx.Button(self, wx.ID_ANY, "Browse")
        output_dir_button = wx.Button(self, wx.ID_ANY, "Browse")

        self.true_ide_edit_field = wx.TextCtrl(self, wx.ID_ANY, "")
        self.adj_ide_edit_field = wx.TextCtrl(self, wx.ID_ANY, "")
        self.output_edit_field = wx.TextCtrl(self, wx.ID_ANY, "")

        self.graph_check_box = wx.CheckBox(self, wx.ID_ANY, "Plot Signals")

        synchronize_signals_button = wx.Button(self, wx.ID_ANY, "Synchronize Signals")

        logo_path = "logo.png"
        bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        logo_path = os.path.join(bundle_dir, logo_path)

        png_logo = wx.Image(logo_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        logo_bitmap = wx.StaticBitmap(self, -1, png_logo, (10, 5), (png_logo.GetWidth(), png_logo.GetHeight()))

        true_ide_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        true_ide_file_sizer.AddSpacer(SPACER_SIZE)
        true_ide_file_sizer.Add(true_ide_browse_button, .2)
        true_ide_file_sizer.AddSpacer(SPACER_SIZE)
        true_ide_file_sizer.Add(self.true_ide_edit_field, 1)
        true_ide_file_sizer.AddSpacer(SPACER_SIZE)

        adj_ide_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        adj_ide_file_sizer.AddSpacer(SPACER_SIZE)
        adj_ide_file_sizer.Add(adj_ide_browse_button, .2)
        adj_ide_file_sizer.AddSpacer(SPACER_SIZE)
        adj_ide_file_sizer.Add(self.adj_ide_edit_field, 1)
        adj_ide_file_sizer.AddSpacer(SPACER_SIZE)

        output_dir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        output_dir_sizer.AddSpacer(SPACER_SIZE)
        output_dir_sizer.Add(output_dir_button, .2)
        output_dir_sizer.AddSpacer(SPACER_SIZE)
        output_dir_sizer.Add(self.output_edit_field, 1)
        output_dir_sizer.AddSpacer(SPACER_SIZE)

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(true_ide_file_label, 0, wx.ALIGN_CENTER)
        main_sizer.Add(true_ide_file_sizer, 0, wx.ALIGN_CENTER | wx.EXPAND)
        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(adj_ide_file_label, 0, wx.ALIGN_CENTER)
        main_sizer.Add(adj_ide_file_sizer, 0, wx.ALIGN_CENTER | wx.EXPAND)
        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(output_dir_path_label, 0, wx.ALIGN_CENTER)
        main_sizer.Add(output_dir_sizer, 0, wx.ALIGN_CENTER | wx.EXPAND)
        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(self.graph_check_box, 0, wx.ALIGN_CENTER)
        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(synchronize_signals_button, 0, wx.ALIGN_CENTER)
        main_sizer.AddSpacer(SPACER_SIZE)
        main_sizer.Add(logo_bitmap, 0, wx.ALIGN_RIGHT)

        # Bind the true signal browse button to it's callback
        self.Bind(
            wx.EVT_BUTTON,
            lambda x: self.file_or_dir_selected(
                self.true_ide_edit_field,
                wx.FileDialog(self, "Select the true signal ide file", "", "", "*.ide", wx.FD_OPEN)),
            true_ide_browse_button)

        # Bind the adjustable signal browse button to it's callback
        self.Bind(
            wx.EVT_BUTTON,
            lambda x: self.file_or_dir_selected(
                self.adj_ide_edit_field,
                wx.FileDialog(self, "Select the adjustable signal ide file", "", "", "*.ide", wx.FD_OPEN)),
            adj_ide_browse_button)

        # Bind the output directory browse button to it's callback
        self.Bind(
            wx.EVT_BUTTON,
            lambda x: self.file_or_dir_selected(self.output_edit_field, wx.DirDialog(self, "Choose directory")),
            output_dir_button)

        # Bind the "Synchronize Signals" button to it's callback
        self.Bind(wx.EVT_BUTTON, lambda e: self.synchronize(), synchronize_signals_button)

        self.SetSizer(main_sizer)
        self.Show()

    @staticmethod
    def file_or_dir_selected(edit_field, browse_dialog):
        if browse_dialog.ShowModal() == wx.ID_OK:
            # Clear the edit field and populate it with the new path
            edit_field.SetValue(browse_dialog.GetPath())

    def check_and_get_user_inputs(self):
        """
        TODO:
         - (maybe) verify we have write access permission for the given output director
        """
        true_ide_path = self.true_ide_edit_field.GetLineText(0)
        adj_ide_path = self.adj_ide_edit_field.GetLineText(0)
        output_path = self.output_edit_field.GetLineText(0)

        create_graphs = self.graph_check_box.IsChecked()

        # Check that all given files and directories are valid
        if not os.path.isfile(true_ide_path):
            wx.MessageBox("The True Signal does not exist!",
                          "Invalid File Location", wx.OK, self)
            return 4*[None]
        if not os.path.isfile(adj_ide_path):
            wx.MessageBox("The Adjustable Signal does not exist!",
                          "Invalid File Location", wx.OK, self)
            return 4*[None]
        if not os.path.isdir(output_path):
            wx.MessageBox("The Output Directory given does not exist!",
                          "Invalid File Location", wx.OK, self)
            return 4*[None]

        return true_ide_path, adj_ide_path, output_path, create_graphs

    def synchronize(self):
        true_ide_path, adj_ide_path, output_path, create_graphs = self.check_and_get_user_inputs()
        if true_ide_path is not None:
            with wx.ProgressDialog(
                    title="Synchronizing Signals",
                    message="Starting synchronization",
                    maximum=7,
                    parent=self,
                    style=wx.PD_APP_MODAL) as progress_bar:

                sync_and_create_new_csv(
                    true_ide_path,
                    adj_ide_path,
                    output_path,
                    progress_callback=ProgressDialogWrapper(progress_bar),
                    show_signal_plots=create_graphs)


if __name__ == "__main__":
    app = wx.App(False)
    frame = SynchronizationWindow(None, "UTick Signal Synchronization")
    app.MainLoop()
