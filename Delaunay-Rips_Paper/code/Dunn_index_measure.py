# Author: Amish Mishra
# Date: December 24, 2021
# README: Calculate Dunn-index vs noise graph


import numpy as np
import tadasets
import matplotlib.pyplot as plt
import Del_Rips as DR
import cechmate as cm
from persim import plot_diagrams, bottleneck
from ripser import ripser


# TODO: compare PDs in the upper triangle of comparison matrix
def intra_cluster_max_dist(cluster):    # returns the size of a given cluster
    max_dist = -1
    for x in cluster:
        for y in cluster:
            dist = bottleneck(x, y)
            if dist > max_dist:
                max_dist = dist
    return max_dist


def inter_cluster_min_dist(cluster1, cluster2): # returns the distance between two clusters
    min_dist = None
    for pd1 in cluster1:
        for pd2 in cluster2:
            dist = bottleneck(pd1, pd2)
            if min_dist is None:
                min_dist = dist
            if dist < min_dist:
                min_dist = dist
    return min_dist


num_per_cluster = 30
num_points = 100
noise = 0.01
hom_class = 2
PD_dataset1 = [None]*num_per_cluster
PD_dataset2 = [None]*num_per_cluster

for i in range(num_per_cluster):
    # Delaunay-Rips
    filtration = DR.build_filtration(tadasets.dsphere(n=num_points, d=2, r=1, noise=noise), hom_dim=hom_class)
    PD_dataset1[i] = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]
    filtration = DR.build_filtration(tadasets.torus(n=num_points, c=2, a=1, noise=noise), hom_dim=hom_class)
    PD_dataset2[i] = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]

all_clusters = [PD_dataset1, PD_dataset2]
min_inter_dist = None
max_intra_dist = -1
for i, C_i in enumerate(all_clusters):
    for j, C_j in enumerate(all_clusters):
        if i != j:  # found different clusters
            inter_dist = inter_cluster_min_dist(C_i, C_j)
            if min_inter_dist is None:
                min_inter_dist = inter_dist
            if inter_dist < min_inter_dist:
                min_inter_dist = inter_dist
    intra_dist = intra_cluster_max_dist(C_i)
    if intra_dist > max_intra_dist:
        max_intra_dist = intra_dist

dunn_index = min_inter_dist/max_intra_dist
print(dunn_index)











# def perturb(data, noise):
#     perturb_vects = np.random.randn(len(data),len(data[0]))
#     mags = np.linalg.norm(perturb_vects, axis=1)
#     for i, mag in enumerate(mags):
#         perturb_vects[i] /= mag*(1/noise) # check that this is actually appying to the whole row. Also, perturb by at MOST noise, not exactly noise
#     perturbed_data = data + perturb_vects
#     return(perturbed_data)


# start_noise = 0.0001
# max_noise = 2
# noise_inc = .1
# pts = 100
# dim = 2
# hom_class = 1
# trials = 5
# del_rips_color = 'b'
# rips_color = 'r'
# alpha_color = 'g'

# data = tadasets.dsphere(n=pts, d=dim, r=1, noise=start_noise)
# # plt.scatter(data[:,0], data[:,1])
# # plt.show()

# # Delaunay-Rips on original data
# filtration = DR.build_filtration(data, dim)
# original_PD_DR  = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]

# # Rips on original data
# original_PD_R = ripser(data, maxdim=dim)['dgms'][hom_class]

# # Alpha on original data
# alpha = cm.Alpha(verbose=False)
# alpha_filtration = alpha.build(2*data)
# original_PD_A = alpha.diagrams(alpha_filtration)[hom_class]
# # plot_diagrams(original_PD_DR, show=True)
# # plot_diagrams(original_PD_R, show=True)
# # plot_diagrams(original_PD_A, show=True)

# noise_arr = np.array(start_noise)
# bott_dist_arr_DR = np.array(0)
# bott_dist_arr_R = np.array(0)
# bott_dist_arr_A = np.array(0)
# for curr_noise in np.arange(start_noise + noise_inc, max_noise, noise_inc):
#     rounded_curr_noise = round(curr_noise, 3)
#     DR_trial_array = [None]*trials
#     R_trial_array = [None]*trials
#     A_trial_array = [None]*trials
#     for t in range(trials):
#         perturbed_data = perturb(data, rounded_curr_noise)
#         # Delaunay-Rips
#         filtration = DR.build_filtration(perturbed_data, dim)
#         dgms_dr = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]
#         bott_dist = bottleneck(original_PD_DR, dgms_dr)
#         DR_trial_array[t] = bott_dist
        

#         # Rips
#         dgms_rips = ripser(perturbed_data, maxdim=dim)['dgms'][hom_class]
#         bott_dist = bottleneck(original_PD_R, dgms_rips)
#         R_trial_array[t] = bott_dist

#         # Alpha
#         alpha = cm.Alpha(verbose=False)
#         alpha_filtration = alpha.build(2*perturbed_data)
#         dgms_alpha = alpha.diagrams(alpha_filtration)[hom_class]
#         bott_dist = bottleneck(original_PD_A, dgms_alpha)
#         A_trial_array[t] = bott_dist

#     bott_dist_arr_DR = np.append(bott_dist_arr_DR, np.median(DR_trial_array))
#     bott_dist_arr_R = np.append(bott_dist_arr_R, np.median(R_trial_array))
#     bott_dist_arr_A = np.append(bott_dist_arr_A, np.median(A_trial_array))
#     plt.boxplot(DR_trial_array, showfliers=False, positions=[rounded_curr_noise], 
#                 widths=noise_inc/5, boxprops=dict(color=del_rips_color), capprops=dict(color=del_rips_color),
#                 whiskerprops=dict(color=del_rips_color), medianprops=dict(color=del_rips_color))
#     plt.boxplot(R_trial_array, showfliers=False, positions=[rounded_curr_noise], 
#                 widths=noise_inc/5, boxprops=dict(color=rips_color), capprops=dict(color=rips_color),
#                 whiskerprops=dict(color=rips_color), medianprops=dict(color=rips_color))
#     plt.boxplot(A_trial_array, showfliers=False, positions=[rounded_curr_noise], 
#                 widths=noise_inc/5, boxprops=dict(color=alpha_color), capprops=dict(color=alpha_color),
#                 whiskerprops=dict(color=alpha_color), medianprops=dict(color=alpha_color))

#     noise_arr = np.append(noise_arr, rounded_curr_noise)

# # Plotting both the curves simultaneously
# # print(bott_dist_arr_A, bott_dist_arr_R, bott_dist_arr_DR)
# plt.plot(noise_arr, bott_dist_arr_DR, color=del_rips_color, label='Del-Rips')
# plt.plot(noise_arr, bott_dist_arr_R, color=rips_color, label='Rips')
# plt.plot(noise_arr, bott_dist_arr_A, color=alpha_color, label='Alpha')

# # Naming the x-axis, y-axis and the whole graph
# plt.xlabel("Noise", fontsize=16)
# plt.xlim([0, max_noise])
# plt.ylabel("Bottleneck Distance", fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Adding legend, which helps us recognize the curve according to it's color
# plt.legend(fontsize=14)

# plt.show()
