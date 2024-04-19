#  This file is part of DeepSpinePreprocessing
#  Copyright (C) 2021 VG-Lab (Visualization & Graphics Lab), Universidad Rey Juan Carlos
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt

from gmrv_utils.plots import colorDict, createColorMap
from gmrv_utils.image import load_tiff_from_folder, multiChannel2SingleChannel
from gmrv_utils.cmpPlot import CmpPlot

from seg_utils.dataLoading import loadTiffStacks
from PyQt5 import Qt

# =============================================================================
# Inicialización
# =============================================================================
path = './testDataGTnewRadius/'
dataSet = 0
clear = False

# =============================================================================
# Carga de todos los datos
# =============================================================================
files2Load = ('LABEL_' + str(dataSet), 'PRED_' + str(dataSet), 'RAW_' + str(dataSet))
data = loadTiffStacks(path, files2Load, False, False)

label = data['LABEL_' + str(dataSet)]
pred = data['PRED_' + str(dataSet)]
raw = data['RAW_' + str(dataSet)]


class SegmentationComparatorView(Qt.QWidget):
    def __init__(self, parent=None, rawImg=None, labelImg=None, predImg=None):
        Qt.QMainWindow.__init__(self, parent)
        self.raw = rawImg
        self.label = labelImg
        self.pred = predImg
        self.vl = Qt.QVBoxLayout(self)
        self.plotData()

    # =============================================================================
    # Plot de restultados
    # =============================================================================
    def plotData(self):
        # =============================================================================
        # Plot de resultados
        # =============================================================================
        cmcmp3 = createColorMap('cmcmp3', colorDict['cmp3'])
        cm3 = createColorMap('cm3', [(0.99, 0.99, 0.99), (0, 0.8, 0), (0, 0.5, 1)])
        cmc = createColorMap('cmc', [(0.95, 1, 0.95, 1), (0, 1, 0, 1)], N=255)
        cmap = (cm3, cm3, cmc, cmcmp3)

        cp = []

        cmpdata = (self.label, self.pred, self.raw, self.pred + self.label * 3)

        title = ["GT", "PRED", "RAW", "CMP"]

        cp.append(CmpPlot(cmpdata, 2, title=title, cmap=cmap, widget=self))


if __name__ == '__main__':
    app = Qt.QApplication([])
    segmentationComparatorView = SegmentationComparatorView(rawImg=raw,
                                                            labelImg=label, predImg=pred)
    segmentationComparatorView.show()
    app.exec_()
