import numpy as np
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pyqtgraph as pg


class AnalysisMW(QMainWindow):
	def __init__(self):
		super().__init__()

		analysis = QWidget()
		self.setCentralWidget(analysis)

		self.buttonlist = []
		self.variables = []
		self.data = []
		self.graph = pg.PlotWidget(title="Plot")
		self.plot = self.graph.plot(x=[1,0],y=[0,1],pen=None, symbol="o")

		buttoncontainerlayout = QHBoxLayout()
		buttoncontainerlayout.addWidget(QLabel("Plot temperature vs..."))
		self.buttonlayout = QHBoxLayout()
		buttoncontainerlayout.addLayout(self.buttonlayout)

		filebutton = QPushButton("Choose File")
		filebutton.clicked.connect(self.chooseFile)

		self.fileloc = QLineEdit()

		filelayout = QHBoxLayout()
		filelayout.addWidget(self.fileloc)
		filelayout.addWidget(filebutton)

		wholelayout = QVBoxLayout()
		wholelayout.addLayout(filelayout)
		wholelayout.addLayout(buttoncontainerlayout)
		wholelayout.addWidget(self.graph)

		analysis.setLayout(wholelayout)

		self.setGeometry(500, 180, 400, 400)
		self.setWindowTitle("Analysis")
		self.show()

	def generateButtons(self, names):
		self.clearButtons()
		for name in names:
			print(name)
			tempbutton = QPushButton(name)
			tempbutton.clicked.connect(self.graphData)
			self.buttonlist.append(tempbutton)
			self.buttonlayout.addWidget(tempbutton)

	def clearButtons(self):
		while self.buttonlayout.count():
			child = self.buttonlayout.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
			self.buttonlist.pop(0)

	def chooseFile(self):
		filename = QFileDialog.getOpenFileName(self, "Open File", os.path.join(os.getcwd(), "data"))

		if filename[0]:
			self.fileloc.setText(filename[0])

			self.variables = np.genfromtxt(filename[0], delimiter=",", skip_header=3, max_rows=1, dtype=str)
			self.data = np.genfromtxt(filename[0], delimiter=",", skip_header=4)
			self.generateButtons(self.variables[1:])
			#print(data[:, 0]) gives temperatures

	def graphData(self, sender):
		buttontext = self.sender().text()
		self.graph.setTitle("{} vs. Temperature".format(buttontext))
		coltoplot = self.buttonlist.index(self.sender()) + 1
		self.plot.setData(x=self.data[:,0],y=self.data[:, coltoplot])



#a = np.genfromtxt(os.path.join(os.getcwd(), "data\corr_20180123-034342.csv"), delimiter=",", skip_header=4)
#print(a)
if __name__ == '__main__':
	app = QApplication(sys.argv)
	bd = AnalysisMW()
	sys.exit(app.exec_())