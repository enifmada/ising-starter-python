import numpy as np
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pyqtgraph as pg
import scipy.optimize

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def hc_func_below(x, tc, alpha, const):
	return alpha * np.log((tc-x)/tc) + const


def hc_func_above(x, tc, beta, const):
	return beta * np.log((x-tc)/tc) + const

def mag_func_below(x, tc, a, beta):
	return a + beta * np.log((tc-x)/tc)

def sus_func_below(x, tc, a, gamma):
	return a + gamma * np.log((tc-x)/tc)

def sus_func_above(x, tc, b, gamma):
	return b + gamma * np.log((x-tc)/tc)


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

		self.mode = None
		self.buttonlist = []
		self.hasstdev = []
		self.variables = []
		self.data = []
		self.valuespertemp = 0
		self.graph = pg.PlotWidget(title=None)
		self.graph.data = None
		self.plot = self.graph.plot(x=[1,0],y=[0,1],pen=None, symbol="o")

		self.funcdict={}

		self.analysisbutton = QPushButton("Fit Data")
		self.analysisbutton.clicked.connect(self.analyzeCurrentData)

		self.headerlayout = QHBoxLayout()


		filebutton = QPushButton("Choose File")
		filebutton.clicked.connect(self.chooseFile)

		self.fileloc = QLineEdit()

		filelayout = QHBoxLayout()
		filelayout.addWidget(self.fileloc)
		filelayout.addWidget(filebutton)

		self.choiceanalysislayout = QHBoxLayout()
		self.choicecontainerlayout = QHBoxLayout()
		self.choiceanalysislayout.addLayout(self.choicecontainerlayout)
		self.choiceanalysislayout.addWidget(self.analysisbutton)
		self.choiceanalysislayout.addStretch(1)

		self.wholelayout = QVBoxLayout()
		self.wholelayout.addLayout(self.headerlayout)
		self.wholelayout.addLayout(filelayout)
		self.wholelayout.addLayout(self.choiceanalysislayout)
		self.wholelayout.addWidget(self.graph)

		analysis.setLayout(self.wholelayout)

		self.setGeometry(500, 180, 400, 400)
		self.setWindowTitle("Analysis")
		self.show()

	def generateButtons(self, names):
		self.clearLayout(self.choicecontainerlayout)
		self.choicecontainerlayout.addWidget(QLabel("Plot temperature vs..."))
		self.buttonlist = []
		for name in names:
			index = names.index(name)
			tempbutton = QPushButton(name)
			tempbutton.clicked.connect(self.graphData)
			self.buttonlist.append(name)
			self.choicecontainerlayout.addWidget(tempbutton)
			if index + 1 < len(names) and names[index+1][-7:] == "Std Dev":
				self.hasstdev.append(True)
				names.pop(index+1)
			else:
				self.hasstdev.append(False)
		self.choicecontainerlayout.addStretch(1)

	def generateCorrComboBox(self):
		self.clearLayout(self.choicecontainerlayout)
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
		self.clearLayout(self.headerlayout)
		for i in range(len(parameters)):
			tempParamBox = ParamBox(parameters[i], values[i])
			self.headerlayout.addLayout(tempParamBox.layout)

	def clearLayout(self, layout):
		while layout.count():
			child = layout.takeAt(0)
			if child.widget() is not None:
				child.widget().deleteLater()
			elif child.layout() is not None:
				self.clearLayout(child.layout())

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
				self.graph.data=None
				if file[:4] == "data":
					self.variables = np.genfromtxt(filename[0], delimiter=",", skip_header=3, max_rows=1, dtype=str)
					self.data = self.data[np.argsort(self.data[:,0])]
					self.generateButtons(self.variables[1:].tolist())
					self.mode = "data"
					# data[:, 0]) gives temperatures
				else:
					sizeindex = self.parameternames.tolist().index("Lattice Size (NxN)")
					self.data = self.data[np.lexsort((self.data[:,1],self.data[:,0]))]
					self.valuespertemp = int(self.parametervalues[sizeindex]/2-1)
					print(self.valuespertemp)
					self.generateCorrComboBox()
					self.mode = "corr"

	def graphData(self, sender):
		buttontext = self.sender().text()
		self.graph.setTitle("{} vs. Temperature".format(buttontext))
		indexofbuttoninbuttons = self.buttonlist.index(buttontext)
		indexofbuttonindata = self.variables[1:].tolist().index(buttontext)
		coltoplot = indexofbuttonindata + 1
		self.graph.clear()
		if buttontext == "Magnetization Mean":
			ydata = np.abs(self.data[:, coltoplot])
		else:
			ydata = self.data[:, coltoplot]
		self.plot = self.graph.plot(x=self.data[:, 0], y=ydata, pen=None, symbol="o")

		if self.hasstdev[indexofbuttoninbuttons]:
			beamwidth = (self.data[1,0]-self.data[0,0])/2
			self.bars = pg.ErrorBarItem(x=self.data[:,0], y=ydata, height = self.data[:, coltoplot+1], left=None, right=None, beam=beamwidth, pen={'color':'b', 'width':1})
			self.graph.addItem(self.bars)
		self.graph.setLabel("left", buttontext)
		self.graph.setLabel("bottom", "Temperature")
		self.graph.data = buttontext

	def graphCorrData(self, index):
		dataloc = index*self.valuespertemp
		self.graph.setTitle("Spatial correlation vs. distance at T = {}".format(str(self.data[dataloc,0])))
		self.graph.clear()
		self.plot = self.graph.plot(x=self.data[dataloc:dataloc+self.valuespertemp,1], y=self.data[dataloc:dataloc+self.valuespertemp,2], pen=None, symbol="o")
		self.graph.setLabel("left", "Spatial correlation")
		self.graph.setLabel("bottom", "Distance")
		self.graph.data = index

	def getHCDivision(self, hc_data):
		high_indices = np.argpartition(hc_data, -4)[-4:]
		high_indices_sorted = np.sort(high_indices)
		run = 1
		best_run = 1
		index_of_best_run = 0
		start_index = 0
		for i in range(0, len(high_indices_sorted)-1):
			if high_indices_sorted[i]+1 in high_indices_sorted:
				run = run + 1
			else:
				if best_run < run:
					index_of_best_run = start_index
					best_run = run
				start_index = i + 1
				run = 1
		if best_run < run:
			index_of_best_run = start_index
			best_run = run
		if best_run > 1:
			print("Peak of size {} found".format(best_run))
			peak_indices = high_indices_sorted[index_of_best_run:index_of_best_run+best_run]
			hc_best = hc_data[peak_indices]
			division_index = 0
			for i in range(len(hc_best)-1):
				if hc_best[i+1] < hc_best[i]:
					break
				division_index = i+1
			return peak_indices[division_index]


		else:
			print("Nothing found!")
			return None

	def getMagDivision(self, mag_data):
		index = 0
		for i in range(0, len(mag_data)):
			index = i
			if mag_data[i] < .1:
				break

		return index

	def analyzeCurrentData(self):
		#tc=2.269
		if self.graph.data:
			if self.mode == "corr":
				pass
			else:
				if self.graph.data == "Heat Capacity":
					index_of_button_in_buttons = self.buttonlist.index(self.graph.data)
					ycol = self.variables.tolist().index(self.graph.data)
					split_index = self.getHCDivision(self.data[:, ycol])
					if split_index:
						hc_data_below = self.data[:split_index, ycol]
						hc_data_above = self.data[split_index:, ycol]
						if self.hasstdev[index_of_button_in_buttons]:
							below_std = self.data[:split_index, ycol+1]
							above_std = self.data[split_index:, ycol+1]
						else:
							below_std = None
							above_std = None
						tc_guess = .5*(self.data[split_index-1, 0]+self.data[split_index,0])
						guess_below = [tc_guess, -1.0, 0]
						guess_above = [tc_guess, -1.0, hc_data_above[-1]]
						popt_below, pcov_below = scipy.optimize.curve_fit(hc_func_below, self.data[:split_index, 0], hc_data_below, p0=guess_below, sigma=below_std, absolute_sigma=False)
						popt_above, pcov_above = scipy.optimize.curve_fit(hc_func_above, self.data[split_index:, 0], hc_data_above, p0=guess_above, sigma=above_std, absolute_sigma=False)
						print("Function below: {0:0.3g} * log ({1:0.3f}-T)/{1:0.3f} + {2:0.3g}".format(popt_below[1], popt_below[0],popt_below[2]))
						print("Tc = {0:0.3f} ± {1:0.3f}".format(popt_below[0], np.sqrt(pcov_below[0,0])))
						print("Function above: {0:0.3g} * log (T-{1:0.3f})/{1:0.3f} + {2:0.3g}".format(popt_above[1],popt_above[0],popt_above[2]))
						print("Tc = {0:0.3f} ± {1:0.3f}".format(popt_above[0], np.sqrt(pcov_above[0, 0])))

						spacing = self.data[1,0]-self.data[0,0]
						below_fitxdata = np.linspace(self.data[0,0]-spacing, self.data[split_index+1,0], num=500)
						below_fitydata = hc_func_below(below_fitxdata, popt_below[0], popt_below[1], popt_below[2])
						self.below_fitplot = self.graph.plot(x=below_fitxdata, y=below_fitydata)

						above_fitxdata = np.linspace(self.data[split_index,0]-spacing/2, self.data[-1,0]+spacing, num=500)
						above_fitydata = hc_func_above(above_fitxdata, popt_above[0], popt_above[1], popt_above[2])
						self.above_fitplot = self.graph.plot(x=above_fitxdata, y=above_fitydata)
				if self.graph.data == "Magnetization Mean":
					index_of_button_in_buttons = self.buttonlist.index(self.graph.data)
					ycol = self.variables.tolist().index(self.graph.data)
					split_index = self.getMagDivision(np.abs(self.data[:, ycol]))
					if split_index:
						mag_data = np.abs(self.data[:split_index, ycol])
						if self.hasstdev[index_of_button_in_buttons]:
							below_std = self.data[:split_index, ycol + 1]
						else:
							below_std = None
						tc_guess = .5 * (self.data[split_index - 1, 0] + self.data[split_index, 0])
						guess_below = [tc_guess, 1.0, .125]
						popt_below, pcov_below = scipy.optimize.curve_fit(mag_func_below, self.data[:split_index, 0], np.log(mag_data), p0=guess_below, sigma=below_std, absolute_sigma=False)
						print("Function below: log(|M|) = {0:0.3g} +  {2:0.3g} * log ({1:0.3f}-T)/{1:0.3f}".format(popt_below[1], popt_below[0], popt_below[2]))
						print("Tc = {0:0.3f} ± {1:0.3f}".format(popt_below[0], np.sqrt(pcov_below[0, 0])))
						print("beta = {0:0.3f} ± {1:0.3f}".format(popt_below[2], np.sqrt(pcov_below[2, 2])))
						tc_real = popt_below[0]

						spacing = self.data[1, 0] - self.data[0, 0]
						below_fitxdata = np.linspace(self.data[0, 0] - spacing, self.data[split_index, 0]-spacing, num=500)
						below_fitydata = np.exp(mag_func_below(below_fitxdata, popt_below[0], popt_below[1], popt_below[2]))
						self.graph.clear()
						self.below_fitplot = self.graph.plot(x=np.log((tc_real-below_fitxdata)/tc_real), y=np.log(below_fitydata))
						self.plot = self.graph.plot(x=np.log((tc_real-self.data[:split_index, 0])/tc_real), y=np.log(mag_data), pen=None, symbol="o")
						self.bars = pg.ErrorBarItem(x=np.log((tc_real-self.data[:split_index, 0])/tc_real), y=np.log(mag_data), height=np.divide(below_std, mag_data),left=None, right=None, beam=spacing/2,pen={'color': 'b', 'width': 1})
						self.graph.addItem(self.bars)
						self.graph.setTitle("log(Magnetization Mean) vs. log((Tc-T)/T))")
						self.graph.setLabel("left", "log(Magnetization Mean)")
						self.graph.setLabel("bottom", "log((Tc-T)/T))")



if __name__ == '__main__':
	app = QApplication(sys.argv)
	bd = AnalysisMW()
	sys.exit(app.exec_())