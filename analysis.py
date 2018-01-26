import numpy as np
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ParamBox(QWidget):
	def __init__(self, parameter, value):
		super().__init__()
		self.layout = QVBoxLayout()
		self.layout.addWidget(QLabel(parameter))
		temp = QLineEdit()
		temp.setText(str(value))
		temp.setReadOnly(True)
		self.layout.addWidget(temp)

class AnalysisMW(QMainWindow):
	def __init__(self):
		super().__init__()

		analysis = QWidget()
		self.setCentralWidget(analysis)

		self.buttonlist = []
		self.hasstdev = []
		self.variables = []
		self.data = []
		self.valuespertemp = 0
		self.graph = pg.PlotWidget(title="Plot")
		self.plot = self.graph.plot(x=[1,0],y=[0,1],pen=None, symbol="o")

		self.headerlayout = QHBoxLayout()


		filebutton = QPushButton("Choose File")
		filebutton.clicked.connect(self.chooseFile)

		self.fileloc = QLineEdit()

		filelayout = QHBoxLayout()
		filelayout.addWidget(self.fileloc)
		filelayout.addWidget(filebutton)

		self.choicecontainerlayout = QHBoxLayout()

		self.wholelayout = QVBoxLayout()
		self.wholelayout.addLayout(self.headerlayout)
		self.wholelayout.addLayout(filelayout)
		self.wholelayout.addLayout(self.choicecontainerlayout)
		self.wholelayout.addWidget(self.graph)

		analysis.setLayout(self.wholelayout)

		self.setGeometry(500, 180, 400, 400)
		self.setWindowTitle("Analysis")
		self.show()

	def generateButtons(self, names):
		self.clearObject(self.choicecontainerlayout)
		self.choicecontainerlayout.addWidget(QLabel("Plot temperature vs..."))
		buttonlayout = QHBoxLayout()
		self.choicecontainerlayout.addLayout(buttonlayout)
		self.choicecontainerlayout.addStretch(1)
		self.buttonlist = []
		for name in names:
			index = names.index(name)
			tempbutton = QPushButton(name)
			tempbutton.clicked.connect(self.graphData)
			self.buttonlist.append(tempbutton)
			buttonlayout.addWidget(tempbutton)
			if index + 1 < len(names) and names[index+1][-7:] == "Std Dev":
				self.hasstdev.append(True)
				names.pop(index+1)
			else:
				self.hasstdev.append(False)

	def generateCorrComboBox(self):
		self.clearObject(self.choicecontainerlayout)
		spacing = self.data[self.valuespertemp,0]-self.data[0,0]
		corrcombobox = QComboBox()
		temp = self.data[0,0]
		while temp <= self.data[-1,0]:
			corrcombobox.addItem(str(temp))
			temp = temp + spacing
		corrcombobox.activated.connect(lambda: self.graphCorrData(corrcombobox.currentIndex()))
		self.choicecontainerlayout.addWidget(QLabel("Plot spatial correlation vs. distance at T = "))
		self.choicecontainerlayout.addWidget(corrcombobox)
		self.choicecontainerlayout.addStretch(1)

		#mintemp = data[0,0]
		#maxtemp = data[-1,0]
		#corrdatadict = {}
		#for i in range(int(data.shape[0]/valuespertemp)):
			#corrdatadict[data[i*valuespertemp,0]] = data[i*valuespertemp:2*i*valuespertemp,1:]

	def generateHeader(self, parameters, values):
		self.clearObject(self.headerlayout)
		for i in range(len(parameters)):
			tempParamBox = ParamBox(parameters[i], values[i])
			self.headerlayout.addLayout(tempParamBox.layout)

	def clearObject(self, object):
		while object.count():
			child = object.takeAt(0)
			if child.widget():
				child.widget().deleteLater()

	def chooseFile(self):
		filename = QFileDialog.getOpenFileName(self, "Open File", os.path.join(os.getcwd(), "data"))

		if filename[0]:
			self.fileloc.setText(filename[0])
			file = os.path.basename(filename[0])
			if file[:4] != "data" and file[:4] != "corr":
				print("Invalid file name!")
			else:
				self.parameternames = np.genfromtxt(filename[0], delimiter=",", max_rows=1, dtype=str)
				self.parametervalues = np.genfromtxt(filename[0], delimiter=",", skip_header=1, max_rows=1)
				self.data = np.genfromtxt(filename[0], delimiter=",", skip_header=4)
				self.generateHeader(self.parameternames.tolist(), self.parametervalues.tolist())
				self.graph.clear()
				self.graph.setTitle(None)
				if file[:4] == "data":
					self.variables = np.genfromtxt(filename[0], delimiter=",", skip_header=3, max_rows=1, dtype=str)
					self.generateButtons(self.variables[1:].tolist())
					# data[:, 0]) gives temperatures
				else:
					sizeindex = self.parameternames.tolist().index("Lattice Size (NxN)")
					self.valuespertemp = int(self.parametervalues[sizeindex]/2-1)
					self.generateCorrComboBox()

	def graphData(self, sender):
		buttontext = self.sender().text()
		self.graph.setTitle("{} vs. Temperature".format(buttontext))
		indexofbuttoninbuttons = self.buttonlist.index(self.sender())
		indexofbuttonindata = self.variables[1:].tolist().index(buttontext)
		coltoplot = indexofbuttonindata + 1
		self.graph.clear()
		self.plot = self.graph.plot(x=self.data[:, 0], y=self.data[:, coltoplot], pen=None, symbol="o")
		if self.hasstdev[indexofbuttoninbuttons]:
			beamwidth = (self.data[1,0]-self.data[0,0])/2
			bars = pg.ErrorBarItem(x=self.data[:,0], y=self.data[:, coltoplot], height = self.data[:, coltoplot+1], left=None, right=None, beam=beamwidth, pen={'color':'b', 'width':1})
			self.graph.addItem(bars)

		self.graph.setLabel("left", buttontext)
		self.graph.setLabel("bottom", "Temperature")

	def graphCorrData(self, index):
		dataloc = index*self.valuespertemp
		self.graph.setTitle("Spatial correlation vs. distance at T = {}".format(str(self.data[dataloc,0])))
		self.graph.clear()
		self.plot = self.graph.plot(x=self.data[dataloc:dataloc+self.valuespertemp,1], y=self.data[dataloc:dataloc+self.valuespertemp,2], pen=None, symbol="o")

#a = np.genfromtxt(os.path.join(os.getcwd(), "data\corr_20180123-034342.csv"), delimiter=",", skip_header=4)
#print(a)
if __name__ == '__main__':
	app = QApplication(sys.argv)
	bd = AnalysisMW()
	sys.exit(app.exec_())