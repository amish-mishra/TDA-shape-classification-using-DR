# Author: Amish Mishra
# Date: January 19, 2022
# README: Generate the PDs of various classes of datasets using different filtrations

import matplotlib.pyplot as plt
import numpy as np
import tadasets
import cechmate as cm
from ripser import ripser
from persim import plot_diagrams
import os
import time

def generate_noisy_data(shape, max_noise, pts):
    # def perturb(data, max_noise):       # local function that perturbs the data by the noise level
    #     perturb_vects = np.random.randn(len(data), len(data[0]))
    #     mags = np.linalg.norm(perturb_vects, axis=1)
    #     for i, mag in enumerate(mags):
    #         noise = np.random.rand()*max_noise # perturb by a length of no more than 'max_noise'
    #         perturb_vects[i] *= noise/mag
    #     perturbed_data = data + perturb_vects
    #     return perturbed_data
    if shape.lower() == "circle":
        data = tadasets.dsphere(n=pts, d=1, r=1, noise=0)
    elif shape.lower() == "sphere":
        data = tadasets.dsphere(n=pts, d=2, r=1, noise=0)
    elif shape.lower() == "torus":
        data = tadasets.torus(n=pts, c=1, a=0.5, noise=0)
    elif shape.lower() == "random":
        data = np.random.rand(pts, 3)
    elif shape.lower() == "clusters":
        centers = np.random.rand(3, 3);  
        a = np.random.randint(np.floor(.1*pts), np.floor(.45*pts))  # random number of points for first cluster, can be at most 45% of all points, but is at least 10%
        b =  np.random.randint(np.floor(.1*(pts-a)), np.floor(.45*(pts-a))) # random number of pts in 2nd cluster, at least 10% remaining, at most 45%
        c = pts-a-b
        data = np.concatenate((np.tile(centers[0,:], (a, 1)), np.tile(centers[1,:], (b, 1)),  np.tile(centers[2,:], (c, 1))))
    elif shape.lower() == "clusters_in_clusters":
        # Set up the 3 clusters with 3 centers in each one
        centers_3 = np.random.rand(3, 3);
        a = np.random.randint(np.floor(.1*pts), np.floor(.45*pts))  # random number of points for first cluster, can be at most 45% of all points, but is at least 10%
        b =  np.random.randint(np.floor(.1*(pts-a)), np.floor(.45*(pts-a))) # random number of pts in 2nd cluster, at least 10% remaining, at most 45%
        c = pts-a-b
        cluster_centers = np.concatenate((np.tile(centers_3[0,:], (3, 1)), np.tile(centers_3[1,:], (3, 1)),  np.tile(centers_3[2,:], (3, 1))))
        perturb_centers = max_noise*np.random.rand(9, 3)
        centers_9 = cluster_centers + perturb_centers # create 9 cluster centers; 3 clusters each in 3 large clusters
        # Create cluster 1 just like in the "clusters" case
        a1 = np.random.randint(np.floor(.1*a), np.floor(.45*a))
        a2 =  np.random.randint(np.floor(.1*(a-a1)), np.floor(.45*(a-a1)))
        a3 = a-a1-a2
        cluster1 = np.concatenate((np.tile(centers_9[0,:], (a1, 1)), np.tile(centers_9[1,:], (a2, 1)),  np.tile(centers_9[2,:], (a3, 1))))
        # Create cluster 2 just like in the "clusters" case
        a1 = np.random.randint(np.floor(.1*b), np.floor(.45*b))
        a2 =  np.random.randint(np.floor(.1*(b-a1)), np.floor(.45*(b-a1)))
        a3 = b-a1-a2
        cluster2 = np.concatenate((np.tile(centers_9[3,:], (a1, 1)), np.tile(centers_9[4,:], (a2, 1)),  np.tile(centers_9[5,:], (a3, 1))))
        # Create cluster 3 just like in the "clusters" case
        a1 = np.random.randint(np.floor(.1*c), np.floor(.45*c))
        a2 =  np.random.randint(np.floor(.1*(c-a1)), np.floor(.45*(c-a1)))
        a3 = c-a1-a2
        cluster3 = np.concatenate((np.tile(centers_9[6,:], (a1, 1)), np.tile(centers_9[7,:], (a2, 1)),  np.tile(centers_9[8,:], (a3, 1))))

        data = np.concatenate((cluster1, cluster2, cluster3))

    perturb_vects = max_noise*np.random.rand(len(data), len(data[0]))
    return data + perturb_vects


def get_pd(filtration_method, data):
    if filtration_method.lower() == "alpha":
        alpha = cm.Alpha(verbose=False)
        filtration = alpha.build(2 * data)  # Alpha goes by radius instead of diameter
        dgms = alpha.diagrams(filtration, verbose=False)
    elif filtration_method.lower() == "rips":
        # only compute homology classes of dimension 1 less than the dimension of the data
        dgm_with_inf = ripser(data, maxdim=(len(data[0])-1))['dgms']
        dgms = dgm_with_inf
        dgms[0] = dgm_with_inf[0][:-1, :]   # remove the H_0 class with infinite persistence
    elif filtration_method.lower() == "del_rips":
        del_rips = cm.DR(verbose=False)
        filtration = del_rips.build(data)
        dgms = del_rips.diagrams(filtration, verbose=False)

    return dgms


# # Testing
# X = generate_noisy_data('clusters_in_clusters', 0.05, 200)   # generate data for a shape
# fig = plt.figure()
#  # syntax for 3-D projection
# ax = plt.axes(projection ='3d')
# if len(X[0]) == 3:
#     ax.scatter(X[:,0], X[:,1], X[:,2])
# else:
#     ax.scatter(X[:,0], X[:,1], np.zeros((len(X), 1)))
# # syntax for plotting
# ax.set_title('3d Scatter plot geeks for geeks')
# plt.show()
# exit()

# Initialize variables
noise_level = 0.05
filtration_func_arr = ["Alpha", "Rips", "Del_Rips"]
k = 2   # maximum homology dimension to output into files
shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]
num_datasets = 50
pts_per_dataset = 500

for i in range(num_datasets):
    for shape_name in shape_name_arr:
        tic = time.time()
        X = generate_noisy_data(shape_name, noise_level, pts_per_dataset)   # generate data for a shape
        for filtration_func in filtration_func_arr: # compute PDs using each filtration for a fixed dataset
            dgm = get_pd(filtration_func, X)
            for j in range(k+1):  # put H_0, H_1,... classes in separate text files
                if j < len(dgm):    # create file path and file name and save the PDs
                    home = os.path.expanduser("~")
                    filepath = f"{home}/Documents/Del_Rips_Paper/research/Delaunay-Rips_Paper/pd_noise_0_05/" \
                               f"{filtration_func}/{shape_name}/"
                    filename = str("PD_"+str(i)+"_"+str(j))
                    np.savetxt(f"{filepath}{filename}.txt", dgm[j])
                    print("Finished making", f"{filepath}{filename}.txt")

