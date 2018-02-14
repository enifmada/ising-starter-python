# import pyqtgraph.examples
# pyqtgraph.examples.run()
import numpy as np

a = np.genfromtxt(r"C:\Users\Adam\Documents\GitHub\ising-starter-python\data\corr_20180212-005310.csv", delimiter=",", skip_header=4, usecols = (0,1,2))
#print(a[np.argsort(a[:,0])])
print(a[:3])
a = a[np.lexsort((a[:,1],a[:,0]))]
print(a)