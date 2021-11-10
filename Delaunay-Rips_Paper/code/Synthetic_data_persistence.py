# Author: Amish Mishra
# Date: November 10, 2021
# README: Del-rips example on some synthetic data

from tadasets.shapes import dsphere
import Del_Rips as DR
import cechmate as cm
import numpy as np
import matplotlib.pyplot as plt
import tadasets
from persim import plot_diagrams



# Sphere
pts = 100
dim = 3
data = tadasets.dsphere(n=pts, d=dim, r=1, noise=0.1)

filtration = DR.build_filtration(data, dim)
dgms_dr = cm.phat_diagrams(filtration, show_inf=True)
plt.title("Persistence Diagram\n"
              "Noisy "+str(dim)+"-sphere ("+str(pts) + " points)")
plot_diagrams(dgms_dr)


# Sphere
plt.figure()
pts = 10000
dim = 3
data = tadasets.torus(n=pts, noise=.1)

filtration = DR.build_filtration(data, dim)
dgms_dr = cm.phat_diagrams(filtration, show_inf=True)
plt.title("Persistence Diagram\n"
              "Noisy torus ("+str(pts) + " points)")
plot_diagrams(dgms_dr)
plt.show()




