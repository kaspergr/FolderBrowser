from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy

class PlotControls(QtWidgets.QWidget):
    def __init__(self, sel_col_func, cmap_func, lim_func, copy_func):
        super(PlotControls, self).__init__()
        self.layout = QtWidgets.QHBoxLayout()
        self.num_col_boxes = 3
        self.num_lim_boxes = 3
        self.sel_col_func = sel_col_func
        self.cmap_func = cmap_func
        self.lim_func = lim_func
        self.copy_func = copy_func
        self.init_col_sel_boxes()
        self.init_cmap_sel()
        self.init_lim_boxes()
        self.init_copy_button()
        self.setLayout(self.layout)

    def reset(self, array_of_text_items):
        assert len(array_of_text_items) == self.num_col_boxes
        for i, box in enumerate(self.col_boxes):
            box.list_of_text_items = array_of_text_items[i]
            prev_text = box.currentText()
            box.clear()
            box.addItems(array_of_text_items[i])
            idx = box.findText(prev_text)
            box.setCurrentIndex(idx)
            min_width = len(max(box.list_of_text_items, key=len)) * 8
            box.view().setMinimumWidth(min_width)
        # All indices must be set in the loop above before we can start
        # assigning lowest unoccupied texts. Otherwise we don't know which
        # texts are unoccupied.
        for box in self.col_boxes:
            if box.currentIndex() == -1:
                self.select_lowest_unoccupied(box)

    def init_col_sel_boxes(self):
        self.col_boxes = [None] * self.num_col_boxes
        for i in range(self.num_col_boxes):
            box = QtWidgets.QComboBox()
            box.activated.connect(self.sel_col_func)
            box.setMaxVisibleItems(80)
            policy_horiz = QSizePolicy.MinimumExpanding
            policy_vert = QSizePolicy.Maximum
            box.setSizePolicy(policy_horiz, policy_vert)
            box.setMinimumWidth(40)
            self.layout.addWidget(box)
            self.col_boxes[i] = box

    def init_cmap_sel(self):
        cmap_sel = QtWidgets.QComboBox()
        cmap_sel.addItems(['Reds', 'Blues_r', 'symmetric'])
        cmap_sel.activated.connect(self.cmap_func)
        policy_horiz = QSizePolicy.MinimumExpanding
        policy_vert = QSizePolicy.Maximum
        cmap_sel.setSizePolicy(policy_horiz, policy_vert)
        cmap_sel.setMinimumWidth(40)
        self.layout.addWidget(cmap_sel)
        self.cmap_sel = cmap_sel

    def init_lim_boxes(self):
        self.lim_boxes = [None] * self.num_lim_boxes
        for i in range(self.num_lim_boxes):
            lim_box = QtWidgets.QLineEdit()
            lim_box.editingFinished.connect(self.lim_func)
            self.layout.addWidget(lim_box)
            self.lim_boxes[i] = lim_box

    def init_copy_button(self):
        copy_button = QtWidgets.QPushButton('C')
        copy_button.clicked.connect(self.copy_func)
        copy_button.setFixedWidth(15)
        self.layout.addWidget(copy_button)
        self.copy_button = copy_button

    def get_sel_texts(self):
        sel_texts = [box.currentText() for box in self.col_boxes]
        return sel_texts

    def select_lowest_unoccupied(self, box):
        """
        Sets the text on box to the text with the lowest index in
        box.list_of_text_items which is not already selected in another box in
        self.col_boxes.
        """
        sel_texts = self.get_sel_texts()
        for i, text in enumerate(box.list_of_text_items):
            if text not in sel_texts:
                box.setCurrentIndex(i)
                return

    def set_text_on_box(self, box_idx, text):
        """
        Potential infinite loop if sel_col_func calls this function.
        """
        box = self.col_boxes[box_idx]
        idx = box.findText(text)
        box.setCurrentIndex(idx)
        self.sel_col_func()