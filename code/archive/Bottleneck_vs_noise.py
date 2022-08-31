# Author: Amish Mishra
# Date: December 10, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a bottlneck vs noise graph for increasing noise on a ground dataset


import numpy as np
import math
import tadasets
import matplotlib.pyplot as plt
import Del_Rips as DR
import cechmate as cm
from persim import plot_diagrams, bottleneck
import time
from ripser import ripser


def perturb(data, noise):
    perturb_vects = np.random.randn(len(data),len(data[0]))
    mags = np.linalg.norm(perturb_vects, axis=1)
    for i, mag in enumerate(mags):
        perturb_vects[i] /= mag*(1/noise)
    perturbed_data = data + perturb_vects
    return(perturbed_data)


start_noise = 0.001
max_noise = 0.2
noise_inc = .01
pts = 200
dim = 2
hom_class = 1


data = tadasets.dsphere(n=pts, d=dim, r=1, noise=start_noise)
# plt.scatter(data[:,0], data[:,1])
# plt.show()

# Delaunay-Rips on original data
filtration = DR.build_filtration(data, dim)
original_PD_DR  = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]

# Rips on original data
original_PD_R = ripser(data, maxdim=dim)['dgms'][hom_class]

# Alpha on original data
alpha = cm.Alpha(verbose=False)
alpha_filtration = alpha.build(2*data)
original_PD_A = alpha.diagrams(alpha_filtration)[hom_class]
# plot_diagrams(original_PD_DR, show=True)
# plot_diagrams(original_PD_R, show=True)
# plot_diagrams(original_PD_A, show=True)

noise_arr = np.array(start_noise)
bott_dist_arr_DR = np.array(0)
bott_dist_arr_R = np.array(0)
bott_dist_arr_A = np.array(0)
for curr_noise in np.arange(noise_inc, max_noise, noise_inc):
    data = perturb(data, curr_noise)

    # Delaunay-Rips
    filtration = DR.build_filtration(data, dim)
    dgms_dr = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]
    bott_dist = bottleneck(original_PD_DR, dgms_dr)
    bott_dist_arr_DR = np.append(bott_dist_arr_DR, bott_dist)

    # Rips
    dgms_rips = ripser(data, maxdim=dim)['dgms'][hom_class]
    bott_dist = bottleneck(original_PD_R, dgms_rips)
    bott_dist_arr_R = np.append(bott_dist_arr_R, bott_dist)

    # Alpha
    alpha = cm.Alpha(verbose=False)
    alpha_filtration = alpha.build(2*data)
    dgms_alpha = alpha.diagrams(alpha_filtration)[hom_class]
    bott_dist = bottleneck(original_PD_A, dgms_alpha)
    bott_dist_arr_A = np.append(bott_dist_arr_A, bott_dist)

    noise_arr = np.append(noise_arr, curr_noise)

# Plotting both the curves simultaneously
# print(bott_dist_arr_A, bott_dist_arr_R, bott_dist_arr_DR)
plt.plot(noise_arr, bott_dist_arr_R, color='r', label='Rips')
plt.plot(noise_arr, bott_dist_arr_DR, color='b', label='Del-Rips')
plt.plot(noise_arr, bott_dist_arr_A, color='g', label='Alpha')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Noise", fontsize=16)
plt.ylabel("Bottleneck Distance", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adding legend, which helps us recognize the curve according to it's color
plt.legend(fontsize=14)

plt.show()
