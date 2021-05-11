""" Module for testing the filter options window """
import sys
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from visualization.filtering.filter_options import FilterOptions
from visualization.filtering.filter_info import FilterInfo
from visualization.plot import MainPage
from visualization.plot import check_args, get_args
from unittest.mock import patch

class TestFilter(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        patch('sys.argv', ["--show","1"])
        args = get_args()
        check_args(args)
        self.parent = MainPage(args, self.app)
        self.filter_info = FilterInfo()
        self.filter_info.do_lp = 0
        self.ui = FilterOptions(self.filter_info, self.parent)

    def test_setup(self):
        self.assertEqual(self.ui.cbox_lp.isChecked(),0)
        self.assertEqual(self.ui.cbox_hp.isChecked(),1)
    
    def test_click_cbox(self):
        self.assertEqual(self.ui.cbox_lp.isChecked(),0)
        QTest.mouseClick(self.ui.cbox_lp, Qt.LeftButton)
        self.assertEqual(self.ui.cbox_lp.isChecked(),1)
    
    def tearDown(self):
        sys.exit()

if __name__ == '__main__':
    unittest.main()