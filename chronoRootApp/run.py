import os
import sys

# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

from interface import Ui_ChronoRootAnalysis
from PyQt5 import QtWidgets

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_ChronoRootAnalysis()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())