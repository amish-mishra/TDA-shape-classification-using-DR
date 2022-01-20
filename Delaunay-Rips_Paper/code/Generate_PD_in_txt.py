# Author: Amish Mishra
# Date: January 19, 2022
# README: Generate the PDs of various classes of datasets using different filtrations

import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
import cechmate as cm


X = np.random.rand(20, 2)
rips = cm.Rips(maxdim=1, verbose=False) #Go up to 1D homology
filtration = rips.build(X)
dgmsrips = rips.diagrams(filtration, verbose=False)

plt.figure()
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.axis('square')
plt.title("Point Cloud")
plt.subplot(122)
plot_diagrams(dgmsrips)
plt.title("Rips Persistence Diagrams")
plt.tight_layout()
plt.show()


alpha = cm.Alpha(verbose=False)
filtration = alpha.build(2*X) # Alpha goes by radius instead of diameter
dgmsalpha = alpha.diagrams(filtration, verbose=False)


plt.figure()
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.axis('square')
plt.title("Point Cloud")
plt.subplot(122)
plot_diagrams(dgmsalpha)
plt.title("Alpha Persistence Diagrams")
plt.tight_layout()
plt.show()

my_filtration = cm.DR(verbose=False)
filtration = my_filtration.build(X) # Alpha goes by radius instead of diameter
dgmsDR = my_filtration.diagrams(filtration, verbose=False)


plt.figure()
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.axis('square')
plt.title("Point Cloud")
plt.subplot(122)
plot_diagrams(dgmsDR)
plt.title("Delaunay-Rips Persistence Diagrams")
plt.tight_layout()
plt.show()