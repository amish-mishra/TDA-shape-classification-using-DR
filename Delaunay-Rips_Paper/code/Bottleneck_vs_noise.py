# Author: Amish Mishra
# Date: December 10, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a bottlneck vs noise graph for increasing noise on a ground dataset


import numpy as np
import tadasets
import matplotlib.pyplot as plt
import Del_Rips as DR
import cechmate as cm
from persim import plot_diagrams, bottleneck
import time
from ripser import ripser

start_noise = 0.001
max_noise = 1
noise_inc = 0.1
pts = 50

# Delaunay-Rips on original data
data = tadasets.dsphere(n=pts, d=1, r=1, noise=start_noise)
filtration = DR.build_filtration(data, 1)
original_PD_DR  = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[0]

# Rips on original data
filtration = DR.build_filtration(data, 1)
original_PD_R = ripser(data, maxdim=1)['dgms'][0]

# Alpha on original data
alpha = cm.Alpha()
alpha_filtration = alpha.build(2*data)
original_PD_A = alpha.diagrams(alpha_filtration)[0]

noise_arr = np.array(start_noise)
bott_dist_arr_DR = np.array(0)
bott_dist_arr_R = np.array(0)
bott_dist_arr_A = np.array(0)
for curr_noise in np.arange(noise_inc, max_noise, noise_inc):
    data = tadasets.dsphere(n=pts, d=1, r=1, noise=curr_noise)

    # Delaunay-Rips
    filtration = DR.build_filtration(data, 1)
    dgms_dr = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[0]
    bott_dist = bottleneck(original_PD_DR, dgms_dr)
    bott_dist_arr_DR = np.append(bott_dist_arr_DR, bott_dist)

    # Rips
    dgms_rips = ripser(data, maxdim=1)['dgms'][0]
    bott_dist = bottleneck(original_PD_R, dgms_rips)
    bott_dist_arr_R = np.append(bott_dist_arr_R, bott_dist)

    # Alpha
    alpha = cm.Alpha()
    alpha_filtration = alpha.build(2*data)
    dgms_alpha = alpha.diagrams(alpha_filtration)[0]
    bott_dist = bottleneck(original_PD_A, dgms_alpha)
    bott_dist_arr_A = np.append(bott_dist_arr_A, bott_dist)

    noise_arr = np.append(noise_arr, curr_noise)

# Plotting both the curves simultaneously
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
