from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore
from customcomboboxes import CustomComboBoxes
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.colors as mcolors


class MplLayout(QtGui.QWidget):
    """
    Contains canvas, toolbar and a customcomboboxes object.
    """
    def __init__(self, statusBar=None):
        super(MplLayout, self).__init__()
        fig = Figure()
        fig.add_subplot(1, 1, 1)
        self.statusBar = statusBar
        self.fig_canvas = FigureCanvasQTAgg(fig)
        self.comboBoxes = CustomComboBoxes(3, self.update_sel_cols, self.update_cmap, self.update_lims)
        self.navi_toolbar = NavigationToolbar2QT(self.fig_canvas, self)
        layout = QtGui.QGridLayout()
        n_rows_canvas = 3
        n_cols_canvas = 8
        for i, box in enumerate(self.comboBoxes.boxes):
            layout.addWidget(box, n_rows_canvas+1, i, 1, 1)
        layout.addWidget(self.comboBoxes.cmap_sel, n_rows_canvas+1, 3, 1, 1)
        for i, lim_box in enumerate(self.comboBoxes.lim_boxes):
            layout.addWidget(self.comboBoxes.lim_boxes[i], n_rows_canvas+1, i+4, 1, 1)
        self.copy_button = QtGui.QPushButton('Copy', self)
        self.copy_button.clicked.connect(self.copy_fig_to_clipboard)
        layout.addWidget(self.copy_button, n_rows_canvas+1, i+5, 1, 1)
        layout.addWidget(self.fig_canvas, 1, 0, n_rows_canvas, n_cols_canvas)
        layout.addWidget(self.navi_toolbar, 0, 0, 1, n_cols_canvas)
        self.setLayout(layout)
        self.none_str = '---'
        self.sel_col_names = self.comboBoxes.get_sel_texts()
        self.cbar = None
        self.image = None
        self.cmaps = ['Reds', 'Blues_r', 'symmetric']
        # Set default colormap.
        self.cmap = 'Reds'
        self.cmap_name = 'Reds'
        self.lims = [None] * 3

    def copy_fig_to_clipboard(self):
        image = QtGui.QPixmap.grabWidget(self.fig_canvas).toImage()
        QtGui.QApplication.clipboard().setImage(image)

    def update_sel_cols(self, new_num=None):
        """
        To maintain a consistent state we must update the plot at the end.
        """
        self.prev_sel_col_names = self.sel_col_names
        self.sel_col_names = self.comboBoxes.get_sel_texts()
        # Try to make 1D plot if '---' is selected in the third comboBox.
        self.plot_is_2D = self.sel_col_names[2] != self.none_str
        self.data_is_1D = self.sweep.dimension == 1
        plot_is_invalid = self.plot_is_2D and self.data_is_1D
        if plot_is_invalid:
            if self.statusBar is not None:
                msg = "You can't do an image plot, since the data is only 1D."
                self.statusBar.showMessage(msg, 2000)
            self.comboBoxes.set_text_on_box(2, self.none_str)
        self.update_plot()

    def reset_and_plot(self, sweep=None):
        if sweep is not None:
            self.sweep = sweep
        raw_col_names = list(self.sweep.data.dtype.names)
        pcol_names = self.sweep.pdata.get_names()
        all_names = raw_col_names + pcol_names
        col3_names = all_names + [self.none_str]
        col_names = [all_names, all_names, col3_names]
        self.comboBoxes.reset(col_names)
        self.update_sel_cols()

    def update_plot(self):
        if self.plot_is_2D: self.update_2D_plot()
        else: self.update_1D_plot()

    def update_1D_plot(self):
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
            self.image = None
        plot_data = self.load_data_for_plot(dim=2)
        for ax in self.fig_canvas.figure.get_axes():
            ax.cla()
            ax.plot(plot_data[0], plot_data[1])
            ax.autoscale_view(True, True, True)
        self.common_plot_update()

    def update_2D_plot(self):
        plot_data = self.load_data_for_plot(dim=3)
        col0_axis = arr_varies_monotonically_on_axis(plot_data[0])
        col1_axis = arr_varies_monotonically_on_axis(plot_data[1])
        if not set((col0_axis, col1_axis)) == set((0, 1)):
            msg = 'Selected columns not valid for image plot. No action taken.'
            self.sel_col_names = self.prev_sel_col_names
            self.statusBar.showMessage(msg)
            return
        col0_lims = [plot_data[0][0,0], plot_data[0][-1,-1]]
        col1_lims = [plot_data[1][0,0], plot_data[1][-1,-1]]
        if col0_axis == 0:
            data_for_imshow = np.transpose(plot_data[2])
        else:
            data_for_imshow = plot_data[2]
        if col0_lims[0] > col0_lims[1]:
            col0_lims.reverse()
            data_for_imshow = np.fliplr(data_for_imshow)
        if col1_lims[0] > col1_lims[1]:
            col1_lims.reverse()
            data_for_imshow = np.flipud(data_for_imshow)
        extent = col0_lims + col1_lims
        fig = self.fig_canvas.figure
        ax = fig.get_axes()[0]
        self.clims = (np.min(data_for_imshow), np.max(data_for_imshow))
        try:
            self.image.set_data(data_for_imshow)
            self.image.set_extent(extent)
        except AttributeError as error:
            ax.cla()
            self.image = ax.imshow(
                data_for_imshow,
                aspect='auto',
                cmap=self.cmap,
                interpolation='none',
                origin='lower',
                extent=extent,
            )
            self.cbar = fig.colorbar(mappable=self.image)
            label = self.sweep.get_label(self.sel_col_names[2])
            self.cbar.set_label(label)
        try:
            self.image.autoscale()
        except ValueError:
            pass
        ax.autoscale_view(True, True, True)
        self.common_plot_update()

    def update_cmap(self, cmap_name=None):
        """
        cmap_name: string corresponding to a built-in matplotlib colormap
              OR 'symmetric' which is defined below.
        """
        if self.image is None:
            return
        if type(cmap_name) is int:
            cmap_name = self.cmaps[cmap_name]
        if cmap_name is None:
            cmap_name = self.cmap_name
        self.cmap_name = cmap_name
        if cmap_name == 'symmetric':
            z_lims = self.lims[2]
            max_abs = np.max(np.abs(z_lims))
            min_val = z_lims[0]
            max_val = z_lims[1]
            if min_val <= 0 <= max_val:
                z_range = max_val - min_val
                n_neg_points = int(abs(min_val)/z_range*100)
                neg_low_limit = 0.5 - abs(min_val)/max_abs/2
                neg_vals = np.linspace(neg_low_limit, 0.5, n_neg_points)
                neg_colors = plt.cm.RdBu_r(neg_vals)
                n_pos_points = int(max_val/z_range*100)
                pos_high_limit = 0.5 + max_val/max_abs/2
                pos_vals = np.linspace(0.5, pos_high_limit, n_pos_points)
                pos_colors = plt.cm.RdBu_r(pos_vals)
                colors = np.vstack((neg_colors, pos_colors))
                cmap = mcolors.LinearSegmentedColormap.from_list('foo', colors)
                self.cmap = cmap
            elif 0 <= min_val <= max_val:
                self.cmap = plt.get_cmap('Reds')
            elif min_val <= max_val <= 0:
                self.cmap = plt.get_cmap('Blues')
        else:
            self.cmap = plt.get_cmap(cmap_name)
        self.image.set_cmap(self.cmap)
        self.cbar.draw_all() # Necessary
        self.fig_canvas.draw()

    def common_plot_update(self):
        ax = self.fig_canvas.figure.get_axes()[0]
        ax.relim()
        xlabel = self.sweep.get_label(self.sel_col_names[0])
        ax.set_xlabel(xlabel)
        ylabel = self.sweep.get_label(self.sel_col_names[1])
        ax.set_ylabel(ylabel)
        if self.cbar is not None:
            label = self.sweep.get_label(self.sel_col_names[2])
            self.cbar.set_label(label)
        ax.set_title(self.sweep.meta['name'], fontsize=10)
        self.custom_tight_layout()
        self.fig_canvas.draw()

    def custom_tight_layout(self):
        # Sometimes we'll get an error:
        # ValueError: bottom cannot be >= top
        # This is a confirmed bug when using tight_layout():
        # https://github.com/matplotlib/matplotlib/issues/5456
        try:
            self.fig_canvas.figure.tight_layout()
        except ValueError:
            msg = ('Title is wider than figure.'
                   'This causes undesired behavior and is a known bug.')
            self.statusBar.showMessage(msg, 2000)

    def parse_lims(self, text):
        lims = text.split(':')
        if len(lims) != 2:
            return (None, None)
        lower_lim = self.conv_to_float_or_None(lims[0])
        upper_lim = self.conv_to_float_or_None(lims[1])
        return (lower_lim, upper_lim)

    def conv_to_float_or_None(self, str):
        try:
            return float(str)
        except ValueError:
            return None

    def update_lims(self):
        ax = self.fig_canvas.figure.get_axes()[0]
        for i, lim_box in enumerate(self.comboBoxes.lim_boxes):
            self.lims[i] = self.parse_lims(lim_box.text())
        if self.image is None:
            data_lims = ax.dataLim.get_points()
        elif self.image is not None:
            tmp = self.image.get_extent()
            data_lims = np.array([[tmp[0], tmp[2]], [tmp[1], tmp[3]]])
        user_lims = np.transpose([self.lims[0], self.lims[1]])
        new_lims = data_lims
        for i in (0,1):
            for j in (0,1):
                if user_lims[i][j] is not None:
                    new_lims[i][j] = user_lims[i][j]
        ax.set_xlim(new_lims[:,0])
        ax.set_ylim(new_lims[:,1])
        self.lims[0] = new_lims[:,0]
        self.lims[1] = new_lims[:,1]
        ax.relim()
        if self.image is None:
            ax.autoscale_view(True, True, True)
        elif self.image is not None:
            new_lims = copy.deepcopy(list(self.clims))
            user_lims = self.lims[2]
            for i in (0,1):
                if user_lims[i] is not None:
                    new_lims[i] = user_lims[i]
            self.lims[2] = new_lims
            self.image.set_clim(new_lims)
            self.update_cmap()
        self.custom_tight_layout()
        self.fig_canvas.draw()

    def load_data_for_plot(self, dim):
        plot_data = [None] * dim
        for i in range(dim):
            try:
                plot_data[i] = self.sweep.data[self.sel_col_names[i]]
            except ValueError:
                plot_data[i] = self.sweep.pdata[self.sel_col_names[i]]
        return plot_data


def arr_varies_monotonically_on_axis(arr):
    for axis in (0,1):
        idx = [0,0]
        idx[axis] = slice(None)
        candidate = arr[idx]
        arr_diff = np.diff(candidate)
        # Check that there are non-zero elements in arr_diff.
        # Otherwise arr is constant.
        if not any(arr_diff):
            continue
        # Check that the elements are the same,
        # i.e., the slope of arr is constant.
        if not np.allclose(arr_diff, arr_diff[0]):
            continue
        # Check that arr consists solely of copies of candidate.
        # First, insert an np.newaxis in candidate so you can subtract it
        # from arr.
        if axis == 0:
            candidate = candidate[...,np.newaxis]
        if not np.allclose(arr, candidate):
            continue
        return axis
    return -1
