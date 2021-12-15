# Author: Amish Mishra
# Date: December 10, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a bottlneck vs noise graph for increasing noise on a ground dataset with median of multiple trials


import numpy as np
import tadasets
import matplotlib.pyplot as plt
import Del_Rips as DR
import cechmate as cm
from persim import plot_diagrams, bottleneck
from ripser import ripser


def perturb(data, noise):
    perturb_vects = np.random.randn(len(data),len(data[0]))
    mags = np.linalg.norm(perturb_vects, axis=1)
    for i, mag in enumerate(mags):
        perturb_vects[i] /= mag*(1/noise)
    perturbed_data = data + perturb_vects
    return(perturbed_data)


start_noise = 0.0001
max_noise = 1.1
noise_inc = .1
pts = 200
dim = 1
hom_class = 1
trials = 10
del_rips_color = 'b'
rips_color = 'r'
alpha_color = 'g'

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
for curr_noise in np.arange(start_noise + noise_inc, max_noise, noise_inc):
    rounded_curr_noise = round(curr_noise, 3)
    DR_trial_array = [None]*trials
    R_trial_array = [None]*trials
    A_trial_array = [None]*trials
    for t in range(trials):
        perturbed_data = perturb(data, rounded_curr_noise)
        # Delaunay-Rips
        filtration = DR.build_filtration(perturbed_data, dim)
        dgms_dr = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]
        bott_dist = bottleneck(original_PD_DR, dgms_dr)
        DR_trial_array[t] = bott_dist
        

        # Rips
        dgms_rips = ripser(perturbed_data, maxdim=dim)['dgms'][hom_class]
        bott_dist = bottleneck(original_PD_R, dgms_rips)
        R_trial_array[t] = bott_dist

        # Alpha
        alpha = cm.Alpha(verbose=False)
        alpha_filtration = alpha.build(2*perturbed_data)
        dgms_alpha = alpha.diagrams(alpha_filtration)[hom_class]
        bott_dist = bottleneck(original_PD_A, dgms_alpha)
        A_trial_array[t] = bott_dist

    bott_dist_arr_DR = np.append(bott_dist_arr_DR, np.median(DR_trial_array))
    bott_dist_arr_R = np.append(bott_dist_arr_R, np.median(R_trial_array))
    bott_dist_arr_A = np.append(bott_dist_arr_A, np.median(A_trial_array))
    plt.boxplot(DR_trial_array, showfliers=False, positions=[rounded_curr_noise], 
                widths=noise_inc/5, boxprops=dict(color=del_rips_color), capprops=dict(color=del_rips_color),
                whiskerprops=dict(color=del_rips_color), medianprops=dict(color=del_rips_color))
    plt.boxplot(R_trial_array, showfliers=False, positions=[rounded_curr_noise], 
                widths=noise_inc/5, boxprops=dict(color=rips_color), capprops=dict(color=rips_color),
                whiskerprops=dict(color=rips_color), medianprops=dict(color=rips_color))
    plt.boxplot(A_trial_array, showfliers=False, positions=[rounded_curr_noise], 
                widths=noise_inc/5, boxprops=dict(color=alpha_color), capprops=dict(color=alpha_color),
                whiskerprops=dict(color=alpha_color), medianprops=dict(color=alpha_color))

    noise_arr = np.append(noise_arr, rounded_curr_noise)

# Plotting both the curves simultaneously
# print(bott_dist_arr_A, bott_dist_arr_R, bott_dist_arr_DR)
plt.plot(noise_arr, bott_dist_arr_DR, color=del_rips_color, label='Del-Rips')
plt.plot(noise_arr, bott_dist_arr_R, color=rips_color, label='Rips')
plt.plot(noise_arr, bott_dist_arr_A, color=alpha_color, label='Alpha')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Noise", fontsize=16)
plt.xlim([0, max_noise])
plt.ylabel("Bottleneck Distance", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adding legend, which helps us recognize the curve according to it's color
plt.legend(fontsize=14)

plt.show()
