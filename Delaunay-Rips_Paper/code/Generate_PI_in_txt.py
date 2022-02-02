# Author: Amish Mishra
# Date: January 26, 2022
# README: Generate the PIs of various PDs from txt files into txt files

import matplotlib.pyplot as plt
import numpy as np
import os
from persim import PersistenceImager


# Initialize variables
noise_level = 0.05
filtration_func_arr = ["Alpha", "Rips", "Del_Rips"]
shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]
num_datasets = 4
pts_per_dataset = 500


# Fit PI attributes to all H_0 classes
pdgms_H0 = []
k = 0   # Hom class to extract
for filtration_func in filtration_func_arr:
    for shape_name in shape_name_arr:
        if shape_name.lower()=="circle" and k==2:   # there are no H_2 persistence pairs for the Circle
            continue
        for i in range(num_datasets):
            path = f"C:\\Users\\amish\Documents\\research\\Delaunay-Rips_Paper\\pd_noise_0_05\\\{filtration_func}\\{shape_name}\\PD_{i}_{k}.txt"
            pd = np.loadtxt(path)
            pdgms_H0.append(pd)

pimgr = PersistenceImager(pixel_size=0.01, birth_range=(0,1))
pimgr.fit(pdgms_H0, skew=True)
pimgr.weight_params = {'n': 1.0}
pimgr.kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
pimgs_H0 = pimgr.transform(pdgms_H0, skew=True)


# Fit PI attributes to all H_1 classes
pdgms_H1 = []
k = 1   # Hom class to extract
for filtration_func in filtration_func_arr:
    for shape_name in shape_name_arr:
        if shape_name.lower()=="circle" and k==2:   # there are no H_2 persistence pairs for the Circle
            continue
        for i in range(num_datasets):
            path = f"C:\\Users\\amish\Documents\\research\\Delaunay-Rips_Paper\\pd_noise_0_05\\\{filtration_func}\\{shape_name}\\PD_{i}_{k}.txt"
            pd = np.loadtxt(path)
            pdgms_H1.append(pd)

pimgr = PersistenceImager(pixel_size=0.01, birth_range=(0,1))
pimgr.fit(pdgms_H1, skew=True)
pimgr.weight_params = {'n': 1.0}
pimgr.kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
pimgs_H1 = pimgr.transform(pdgms_H1, skew=True)


# Fit PI attributes to all H_2 classes
pdgms_H2 = []
k = 0   # Hom class to extract
for filtration_func in filtration_func_arr:
    for shape_name in shape_name_arr:
        if shape_name.lower()=="circle" and k==2:   # there are no H_2 persistence pairs for the Circle
            continue
        for i in range(num_datasets):
            path = f"C:\\Users\\amish\Documents\\research\\Delaunay-Rips_Paper\\pd_noise_0_05\\\{filtration_func}\\{shape_name}\\PD_{i}_{k}.txt"
            pd = np.loadtxt(path)
            pdgms_H2.append(pd)

pimgr = PersistenceImager(pixel_size=0.01, birth_range=(0,1))
pimgr.fit(pdgms_H2, skew=True)
pimgr.weight_params = {'n': 1.0}
pimgr.kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
pimgs_H2 = pimgr.transform(pdgms_H2, skew=True)




fig, axs = plt.subplots(1, 3, figsize=(10,5))

axs[0].set_title("Original Diagram")
pimgr.plot_diagram(pdgms_H1[15], skew=False, ax=axs[0])

axs[1].set_title("Birth-Persistence\nCoordinates")
pimgr.plot_diagram(pdgms_H1[15], skew=True, ax=axs[1])

axs[2].set_title("Persistence Image")

pimgr.plot_image(pimgs_H1[15], ax=axs[2])

plt.tight_layout()
plt.show()


exit()



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
'''