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
def intra_cluster_max_dist(cluster):    # returns the size of a given cluster by taking the max distance between pairs
    max_dist = -1
    for x in cluster:
        for y in cluster:
            dist = bottleneck(x, y)
            if dist > max_dist:
                max_dist = dist
    return max_dist


def intra_cluster_mean_dist(cluster):    # returns the size of a given cluster by finding the mean distance between pairs
    size = len(cluster)
    sum = 0
    for i in range(size):
        for j in range(i, size):
            if i != j:
                sum += bottleneck(cluster[i], cluster[j]) 
    return sum


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


def dunn_index(all_clusters, intra_cluster_method):
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
        intra_dist = intra_cluster_method(C_i)
        if intra_dist > max_intra_dist:
            max_intra_dist = intra_dist
    dunn_index = min_inter_dist/max_intra_dist
    return dunn_index


# Initialize variables
start_noise = 0.001
max_noise = 0.55
noise_inc = .05
num_per_cluster = 20
num_points = 200
hom_class = 2
del_rips_color = 'b'
rips_color = 'r'
alpha_color = 'g'
intra_cluster_method = intra_cluster_max_dist

noise_arr = np.empty(0)
dunn_index_arr_DR = np.empty(0)
dunn_index_arr_R = np.empty(0)
dunn_index_arr_A = np.empty(0)

for curr_noise in np.arange(start_noise + noise_inc, max_noise, noise_inc):
    rounded_curr_noise = round(curr_noise, 3)
    print(rounded_curr_noise)
    PD_dataset1 = [[None for i in range(num_per_cluster)] for j in range(3)]
    PD_dataset2 = [[None for i in range(num_per_cluster)] for j in range(3)]
    for i in range(num_per_cluster):
        data1 = tadasets.dsphere(n=num_points, d=2, r=1, noise=curr_noise)
        data2 = tadasets.torus(n=num_points, c=2, a=1, noise=curr_noise)

        # Delaunay-Rips
        filtration = DR.build_filtration(data1, hom_dim=hom_class)
        PD_dataset1[0][i] = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]
        filtration = DR.build_filtration(data2, hom_dim=hom_class)
        PD_dataset2[0][i] = cm.phat_diagrams(filtration, show_inf=True, verbose=False)[hom_class]

        # Rips
        PD_dataset1[1][i] = ripser(data1, maxdim=hom_class)['dgms'][hom_class]
        PD_dataset2[1][i] = ripser(data2, maxdim=hom_class)['dgms'][hom_class]

        # Alpha
        alpha = cm.Alpha(verbose=False)
        alpha_filtration = alpha.build(2*data1)
        PD_dataset1[2][i] = alpha.diagrams(alpha_filtration)[hom_class]
        alpha = cm.Alpha(verbose=False)
        alpha_filtration = alpha.build(2*data2)
        PD_dataset2[2][i] = alpha.diagrams(alpha_filtration)[hom_class]

    all_clusters_DR = [PD_dataset1[0], PD_dataset2[0]]
    all_clusters_R = [PD_dataset1[1], PD_dataset2[1]]
    all_clusters_A = [PD_dataset1[2], PD_dataset2[2]]

    # Calculate Dunn-index
    dunn_index_DR = dunn_index(all_clusters_DR, intra_cluster_method)
    dunn_index_R = dunn_index(all_clusters_R, intra_cluster_method)
    dunn_index_A = dunn_index(all_clusters_A, intra_cluster_method)

    noise_arr = np.append(noise_arr, rounded_curr_noise)
    dunn_index_arr_DR = np.append(dunn_index_arr_DR, dunn_index_DR)
    dunn_index_arr_R = np.append(dunn_index_arr_R, dunn_index_R)
    dunn_index_arr_A = np.append(dunn_index_arr_A, dunn_index_A)

plt.plot(noise_arr, dunn_index_arr_DR, color=del_rips_color, label='Del-Rips')
plt.plot(noise_arr, dunn_index_arr_R, color=rips_color, label='Rips')
plt.plot(noise_arr, dunn_index_arr_A, color=alpha_color, label='Alpha')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Noise", fontsize=16)
plt.xlim([0, max_noise])
plt.ylabel("Dunn Index", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adding legend, which helps us recognize the curve according to it's color
plt.legend(fontsize=14)

plt.show()
