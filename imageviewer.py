#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:20:50 2017

@author: joe
"""

import sys
from PyQt5 import QtGui, QtWidgets
from imageviewer_gui import Ui_MainWindow
import data_util as d_util


class ImageViewer(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.setupUi(self)
        self.comboBox_predictor_model.addItems(['Nothing is selected',
                                               'ir_rbg_boundry'])
        self.comboBox_image_id.addItems(d_util.get_list_image_id())
        self.checkBox_tree_band_r.stateChanged.connect(self.cb_3_bnd_r_changed)
        self.checkBox_tree_band_b.stateChanged.connect(self.cb_3_bnd_b_changed)
        self.checkBox_tree_band_g.stateChanged.connect(self.cb_3_bnd_g_changed)
        #checkBox_tree_band_rbg
        self.checkBox_tree_band_rbg.stateChanged.connect(self.cb_3_bnd_rbg_changed)

        self.checkBox_a_1.stateChanged.connect(self.checkbox_a_1_changed)
        self.checkBox_a_2.stateChanged.connect(self.checkbox_a_2_changed)
        self.checkBox_a_3.stateChanged.connect(self.checkbox_a_3_changed)
        self.checkBox_a_4.stateChanged.connect(self.checkbox_a_4_changed)
        self.checkBox_a_5.stateChanged.connect(self.checkbox_a_5_changed)
        self.checkBox_a_6.stateChanged.connect(self.checkbox_a_6_changed)
        self.checkBox_a_7.stateChanged.connect(self.checkbox_a_7_changed)
        self.checkBox_a_8.stateChanged.connect(self.checkbox_a_8_changed)

        self.checkBox_m_1.stateChanged.connect(self.checkbox_m_1_changed)
        self.checkBox_m_2.stateChanged.connect(self.checkbox_m_2_changed)
        self.checkBox_m_3.stateChanged.connect(self.checkbox_m_3_changed)
        self.checkBox_m_4.stateChanged.connect(self.checkbox_m_4_changed)
        self.checkBox_m_5.stateChanged.connect(self.checkbox_m_5_changed)
        self.checkBox_m_6.stateChanged.connect(self.checkbox_m_6_changed)
        self.checkBox_m_7.stateChanged.connect(self.checkbox_m_7_changed)
        self.checkBox_m_8.stateChanged.connect(self.checkbox_m_8_changed)

        self.checkBox_p.stateChanged.connect(self.checkbox_p_changed)

        self.lineEdit_image_search.textChanged.connect(self.filter_image_ids)
        self.checkBox_polygon.stateChanged.connect(self.cb_pol_overlay_changed)

        self.comboBox_image_id.currentIndexChanged.connect(self.combo_image_id_changed)
        self.checkBox_large_building_1.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_res_building_1.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_non_res_building_1.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_misc_small_struct_2.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_good_road_3.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_poor_dirt_cart_trl_4.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_footpath_trail_4.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_woodlands_5.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_hedgerows_5.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_group_trees_5.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_standalone_trees_5.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_cont_plow_crop_6.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_row_crop_6.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_waterways_7.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_standing_water_8.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_large_vehicle_9.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_small_vehicle_10.stateChanged.connect(self.determine_selected_classes)
        self.checkBox_motorbike_10.stateChanged.connect(self.determine_selected_classes)
        self.selected_classes = []
        
        self.comboBox_predictor_model.currentIndexChanged.connect(self.combo_predictor_model_changed)

    def combo_predictor_model_changed(self):
        ret = (str(self.comboBox_predictor_model.currentText()))
        if ret == 'ir_rbg_boundry':
            d_util.boundries(str(self.comboBox_image_id.currentText()))
            self.load_image_to_gui()

    def determine_selected_classes(self):
        del self.selected_classes[:]
        if self.checkBox_large_building_1.isChecked() == 1:
            self.selected_classes.append(1)
        if self.checkBox_res_building_1.isChecked() == 1:
            self.selected_classes.append(1)
        if self.checkBox_non_res_building_1.isChecked() == 1:
            self.selected_classes.append(1)
        if self.checkBox_misc_small_struct_2.isChecked() == 1:
            self.selected_classes.append(2)
        if self.checkBox_good_road_3.isChecked() == 1:
            self.selected_classes.append(3)
        if self.checkBox_poor_dirt_cart_trl_4.isChecked() == 1:
            self.selected_classes.append(4)
        if self.checkBox_footpath_trail_4.isChecked() == 1:
            self.selected_classes.append(4)
        if self.checkBox_woodlands_5.isChecked() == 1:
            self.selected_classes.append(5)
        if self.checkBox_hedgerows_5.isChecked() == 1:
            self.selected_classes.append(5)
        if self.checkBox_group_trees_5.isChecked() == 1:
            self.selected_classes.append(5)
        if self.checkBox_standalone_trees_5.isChecked() == 1:
            self.selected_classes.append(5)
        if self.checkBox_cont_plow_crop_6.isChecked() == 1:
            self.selected_classes.append(6)
        if self.checkBox_row_crop_6.isChecked() == 1:
            self.selected_classes.append(6)
        if self.checkBox_waterways_7.isChecked() == 1:
            self.selected_classes.append(7)
        if self.checkBox_standing_water_8.isChecked() == 1:
            self.selected_classes.append(8)
        if self.checkBox_large_vehicle_9.isChecked() == 1:
            self.selected_classes.append(9)
        if self.checkBox_small_vehicle_10.isChecked() == 1:
            self.selected_classes.append(10)
        if self.checkBox_motorbike_10.isChecked() == 1:
            self.selected_classes.append(10)
        self.filter_image_ids()


    def combo_image_id_changed(self):
        ret = d_util.get_classes_in_image(str(self.comboBox_image_id.currentText()))
        msg = ''
        for item in ret:
            msg += str(item) + '\n'
        self.textEdit_statisics_output.setText(msg)

    def cb_pol_overlay_changed(self):
        print("inside cb_pol_overlay_changed.")
        print("set(self.selected_classes)", self.selected_classes)
        print("str(self.comboBox_image_id.currentText()", str(self.comboBox_image_id.currentText()))
        
        d_util.overlay_polygons(str(self.comboBox_image_id.currentText()),
                                    set(self.selected_classes))
        self.load_image_to_gui()

    def filter_image_ids(self):
        self.comboBox_image_id.clear()
        unfilt_img = d_util.get_list_image_id()
        filt_img = (self.lineEdit_image_search.text())
        new_img_ids = [img_id for img_id in unfilt_img if filt_img in img_id]
        if not self.selected_classes:
            self.comboBox_image_id.addItems(new_img_ids)
        else:
            filt_class = d_util.get_images_with_classes(self.selected_classes)
            intersect = set(filt_class).intersection(new_img_ids)
            self.comboBox_image_id.addItems(intersect)
            


    def checkbox_p_changed(self):
        d_util.get_p_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_1_changed(self):
        d_util.get_m_1_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_2_changed(self):
        d_util.get_m_2_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_3_changed(self):
        d_util.get_m_3_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_4_changed(self):
        d_util.get_m_4_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_5_changed(self):
        d_util.get_m_5_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_6_changed(self):
        d_util.get_m_6_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_7_changed(self):
        d_util.get_m_7_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_m_8_changed(self):
        d_util.get_m_8_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_1_changed(self):
        d_util.get_a_1_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_2_changed(self):
        d_util.get_a_2_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_3_changed(self):
        d_util.get_a_3_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_4_changed(self):
        d_util.get_a_4_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_5_changed(self):
        d_util.get_a_5_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_6_changed(self):
        d_util.get_a_6_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_7_changed(self):
        d_util.get_a_7_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def checkbox_a_8_changed(self):
        d_util.get_a_8_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def cb_3_bnd_r_changed(self):
        d_util.get_red_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def cb_3_bnd_b_changed(self):
        d_util.get_blue_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def cb_3_bnd_g_changed(self):
        d_util.get_green_image(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def cb_3_bnd_rbg_changed(self):
        d_util.get_rbg_image_all(str(self.comboBox_image_id.currentText()))
        self.load_image_to_gui()

    def load_image_to_gui(self):
        pixmap = QtGui.QPixmap('testplot.png')
        self.label_image_viewer.setScaledContents(True)
        self.label_image_viewer.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())
