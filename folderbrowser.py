import sys
import os
import platform
import subprocess
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QDesktopWidget, QShortcut
from PyQt5.QtGui import QKeySequence
from filelistwidget import FileList
from sweep import Sweep
from mpllayout import MplLayout
from customdockwidget import CustomDockWidget
from textforcopying import TextForCopying


def show_loading(func):
    def func_wrapper(self, *args, **kwargs):
        self.statusBar.showMessage('Loading...')
        func(self, *args, **kwargs)
        self.statusBar.showMessage('')
    return func_wrapper


class FolderBrowser(QMainWindow):
    def __init__(self, n_layouts, dir_path, name_func_dict,
                 window_title='FolderBrowser'):
        super().__init__()
        self.n_layouts = n_layouts
        self.dir_path = dir_path
        self.name_func_dict = name_func_dict
        self.date_stamp = None
        self.sweep_name = None
        self.setWindowTitle(window_title)
        self.dock_widgets = []
        self.init_statusbar()
        self.init_mpl_layouts()
        self.init_file_list()
        self.setDockNestingEnabled(True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.set_hotkeys()
        self.set_icon()
        self.show()

    @show_loading
    def set_new_sweep(self, file_list_widget=None):
        file_list_item = self.file_list.currentItem()
        sweep_path = file_list_item.data(QtCore.Qt.UserRole)
        self.sweep = Sweep(sweep_path)
        self.sweep.set_pdata(self.name_func_dict)
        title = self.compose_title(self.sweep, sweep_path)
        for mpl_layout in self.mpl_layouts:
            mpl_layout.set_title(title)
            mpl_layout.reset_and_plot(self.sweep)
        self.sweep_path = sweep_path

    def init_statusbar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

    def init_mpl_layouts(self):
        self.mpl_layouts = [None] * self.n_layouts
        for i in range(self.n_layouts):
            self.mpl_layouts[i] = MplLayout(statusBar=self.statusBar,
                                            parent=self)
        for i, mpl_layout in enumerate(self.mpl_layouts):
            title = 'Plot {}'.format(i)
            dock_widget = CustomDockWidget(title, self)
            dock_widget.setWidget(mpl_layout)
            dock_widget.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
            self.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock_widget)
            self.dock_widgets.append(dock_widget)
        self.set_active_layout(self.mpl_layouts[0])

    def init_file_list(self):
        self.file_list = FileList(self.dir_path)
        self.file_list.itemClicked.connect(self.set_new_sweep)
        dock_widget = QDockWidget('Browser', self)
        dock_widget.setWidget(self.file_list)
        dock_widget.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_widget)
        self.dock_widgets.append(dock_widget)

    def reload_file_list(self):
        self.file_list.reload_items()

    def set_active_layout(self, layout):
        try:
            inactive_str = 'background-color: 10; border: none'
            self.active_layout.navi_toolbar.setStyleSheet(inactive_str)
        except AttributeError:
            pass
        active_str = 'background-color: lightblue; border: none'
        layout.navi_toolbar.setStyleSheet(active_str)
        self.active_layout = layout

    def set_hotkeys(self):
        self.copy_fig_hotkey = QShortcut(QKeySequence('Ctrl+c'), self)
        self.copy_fig_hotkey.activated.connect(self.copy_active_fig)
        self.open_folder_hotkey = QShortcut(QKeySequence('Ctrl+Shift+o'), self)
        self.open_folder_hotkey.activated.connect(self.open_folder)
        self.open_folder_hotkey = QShortcut(QKeySequence('F5'), self)
        self.open_folder_hotkey.activated.connect(self.reload_file_list)
        self.open_folder_hotkey = QShortcut(QKeySequence('Ctrl+t'), self)
        self.open_folder_hotkey.activated.connect(self.show_text_for_copying)
        self.open_folder_hotkey = QShortcut(QKeySequence('Ctrl+w'), self)
        self.open_folder_hotkey.activated.connect(self.close)

    def copy_active_fig(self):
        self.active_layout.copy_fig_to_clipboard()
        active_dock_widget = self.active_layout.parentWidget()
        title = active_dock_widget.windowTitle()
        msg = 'Figure in ' + title + ' copied to clipboard'
        self.statusBar.showMessage(msg, 1000)

    def open_folder(self):
        if platform.system() != 'Windows':
            err_msg = 'Open folder only implemented on Windows'
            self.statusBar.showMessage(err_msg)
            return
        norm_path = os.path.normpath(self.sweep_path)
        cmd = ['explorer', norm_path]
        subprocess.Popen(cmd)

    def show_text_for_copying(self):
        lay = self.active_layout
        title = lay.title.replace('\n', ' ')
        date_stamp = self.date_stamp
        name = self.sweep_name
        diag = TextForCopying(title, date_stamp, name, *lay.labels)
        diag.setWindowModality(QtCore.Qt.ApplicationModal)
        diag.setModal(True)
        diag.exec_()

    def set_name_func_dict(self, name_func_dict):
        self.name_func_dict = name_func_dict

    def compose_title(self, sweep, sweep_path):
        self.sweep_name = sweep.meta['name']
        self.date_stamp = os.path.basename(sweep_path)
        return self.date_stamp + '\n' + self.sweep_name

    def set_icon(self):
        app_icon = QtGui.QIcon()
        icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        for size in (16, 24, 32, 48, 256):
            fname = '{}x{}.png'.format(size, size)
            fpath = os.path.join(icons_dir, fname)
            app_icon.addFile(fpath, QtCore.QSize(size,size))
        self.setWindowIcon(app_icon)
