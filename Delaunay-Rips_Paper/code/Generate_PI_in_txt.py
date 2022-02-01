# Author: Amish Mishra
# Date: January 26, 2022
# README: Generate the PIs of various PDs from txt files into txt files

import matplotlib.pyplot as plt
import numpy as np
import os
from persim import PersistenceImager

filtration_func = "Rips"
shape_name = "Circle"
i = 0
j = 0

pd = np.loadtxt("C:\\Users\\amish\Documents\\research\\Delaunay-Rips_Paper\\pd_noise_0_05\\\Rips\\Circle\\PD_24_1.txt")

print(pd)

# Printing a PersistenceImager() object will print its defining attributes
pimgr = PersistenceImager(pixel_size=0.2, birth_range=(0,1))
# PersistenceImager() attributes can be adjusted at or after instantiation.
# Updating attributes of a PersistenceImager() object will automatically update all other dependent attributes.
pimgr.pixel_size = 0.01
# pimgr.birth_range = (0, 0.2)
# The `fit()` method can be called on one or more (*,2) numpy arrays to automatically determine the miniumum birth and
# persistence ranges needed to capture all persistence pairs. The ranges and resolution are automatically adjusted to
# accomodate the specified pixel size.
pimgr.fit(pd, skew=True)
# The `transform()` method can then be called on one or more (*,2) numpy arrays to generate persistence images from diagrams.
# The option `skew=True` specifies that the diagrams are currently in birth-death coordinates and must first be transformed
# to birth-persistence coordinates.
pimgr.weight_params = {'n': 1.0}
pimgr.kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
pimgs = pimgr.transform(pd, skew=True)
# The `plot_diagram()` and `plot_image()` methods can be used to visualize persistence diagrams and images
fig, axs = plt.subplots(1, 3, figsize=(10,5))

axs[0].set_title("Original Diagram")
pimgr.plot_diagram(pd, skew=False, ax=axs[0])

axs[1].set_title("Birth-Persistence\nCoordinates")
pimgr.plot_diagram(pd, skew=True, ax=axs[1])

axs[2].set_title("Persistence Image")

pimgr.plot_image(pimgs, ax=axs[2])
print(pimgs)

plt.tight_layout()
plt.show()


''''
Motta notes:
- Put all H_k pds in a k_dgm array and run the .fit transformer on them to set the scale of the diagram
- 
'''