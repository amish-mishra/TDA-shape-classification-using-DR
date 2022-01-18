# Author: Amish Mishra
# Date: January 18, 2022
# README: Classify 6 data sets using Del-rips, Rips, and Alpha
# Don't concatenate dgms and then compute persistent images; compute images from H_0 and and H_1 and then concatenate

import numpy as np
import matplotlib.pyplot as plt
from persim import PersImage
from persim import PersistenceImager
import os

# Folder Path
path = "/home/amishra/Documents/Del_Rips_Paper/research/Delaunay-Rips_Paper/pd_gauss_0_05"

# Change the directory
os.chdir(path)

circle_pds = [None]*50
clusters_pds = [None]*50
clusters_in_clusters_pds = [None]*50
random_data_pds = [None]*50
sphere_pds = [None]*50
torus_pds = [None]*50


# iterate through all files
for folder in os.listdir(): # iterate through folders
    os.chdir(f"{path}/{folder}")
    i = 0
    for file in os.listdir():   # iterate through files
        # Check whether file is in the desired format or not
        if file.endswith(".persistence"):
            file_path = f"{path}/{folder}/{file}"
            if folder == "Circle":
                circle_pds[i] = np.loadtxt(file_path, skiprows=2, usecols=[1, 2], dtype=float)
            elif folder == "Clusters":
                clusters_pds[i] = np.loadtxt(file_path,  skiprows=2, usecols=[1, 2], dtype=float)
            elif folder == "Clusters within Clusters":
                clusters_in_clusters_pds[i] = np.loadtxt(file_path, skiprows=2, usecols=[1, 2], dtype=float)
            elif folder == "Random Cloud":
                random_data_pds[i] = np.loadtxt(file_path, skiprows=2, usecols=[1, 2], dtype=float)
            elif folder == "Sphere":
                sphere_pds[i] = np.loadtxt(file_path, skiprows=2, usecols=[1, 2], dtype=float)
            elif folder == "Torus":
                torus_pds[i] = np.loadtxt(file_path, skiprows=2, usecols=[1, 2], dtype=float)
            i += 1


pimgr = PersistenceImager(pixel_size=1)
pimgr.fit(circle_pds[0])

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
pimgr.plot_diagram(circle_pds[0], skew=True, ax=axs[0])
axs[0].set_title('Diagram', fontsize=16)

pimgr.plot_image(pimgr.transform(circle_pds[0]), ax=axs[1])
axs[1].set_title('Pixel Size: 1', fontsize=16)

pimgr.pixel_size = 0.1
pimgr.plot_image(pimgr.transform(circle_pds[0]), ax=axs[2])
axs[2].set_title('Pixel Size: 0.1', fontsize=16)

plt.tight_layout()
plt.show()
print("done")

