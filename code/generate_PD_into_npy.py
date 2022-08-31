# Author: Amish Mishra
# Date: February 10, 2022
# README: Generate the PDs of various classes of datasets using different filtrations and save output as .npy

import matplotlib.pyplot as plt
import numpy as np
import tadasets
import cechmate as cm
from ripser import ripser
from persim import plot_diagrams
import os
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def generate_noisy_data(shape, max_noise, pts):
    '''
    generate_noisy_data generates data on a specified shape type, perturbs it, and scales it down to have diameter 1
    '''
    def perturb(data, max_noise):       # local function that perturbs the data by the noise level
        perturb_vects = np.random.randn(len(data), len(data[0]))
        mags = np.linalg.norm(perturb_vects, axis=1)
        for i, mag in enumerate(mags):
            noise = np.random.rand()*max_noise # perturb by a length of no more than 'max_noise'
            perturb_vects[i] *= noise/mag
        return data + perturb_vects
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
        dist_matrix = cdist(centers, centers, metric='euclidean')
        centers = centers/dist_matrix.max() # scale centers to have diam 1 here because it is faster to do it now
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
        dist_matrix = cdist(centers_9, centers_9, metric='euclidean')
        centers_9 = centers_9/dist_matrix.max() # scale centers to have diam 1 here because it is faster to do it now

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
    data = data/get_diameter(data, shape)  # scale data down to have diameter 1
    return perturb(data, max_noise)


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


def get_diameter(X, shape):
    if shape.lower() in ["clusters", "clusters_in_clusters"]: # this has been handled when clusters were generated
        # ConvexHull will not work for 3 clusters of many repeated points because they span a plane, not a 3-D space
        return 1      
    else:
        hull = ConvexHull(X) # Find a convex hull in O(N log N)
        hullpoints = X[hull.vertices,:] # Extract the points forming the hull
        # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        return hdist.max()
        # Get the farthest apart points
        # bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        # print([hullpoints[bestpair[0]], hullpoints[bestpair[1]]])


def main(pd_directory, noise, num_datasets, pts_per_dataset):
    # Initialize variables
    home = os.path.expanduser("~")
    basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/{pd_directory}/"
    noise_level = noise
    filtration_func_arr = ["Alpha", "Rips", "Del_Rips"]
    k = 2   # maximum homology dimension to output into files
    shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]


    # exit()  # this is here so this file doesn't accidentally run and overwrite the previous data

    for i in range(num_datasets):
        for shape_name in shape_name_arr:
            X = generate_noisy_data(shape_name, noise_level, pts_per_dataset)   # generate data for a shape
            for filtration_func in filtration_func_arr: # compute PDs using each filtration for a fixed dataset
                path = f"{basefilepath}{filtration_func}/{shape_name}/"
                dgm = get_pd(filtration_func, X)
                for j in range(k+1):  # put H_0, H_1,... classes in separate text files
                    if j < len(dgm):    # create file path and file name and save the PDs
                        filename = str("PD_"+str(i)+"_"+str(j))
                        np.save(f"{path}{filename}.npy", dgm[j])
                        print("Finished making", f"{path}{filename}.npy")


if __name__ == '__main__':
    # directory_arr = ['pd_noise_0_05', 'pd_noise_0_10', 'pd_noise_0_15', 'pd_noise_0_20', 'pd_noise_0_25',
    #                  'pd_noise_0_30', 'pd_noise_0_35', 'pd_noise_0_45', 'pd_noise_0_50']
    # noise_arr = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50]
    directory_arr = ['pd_noise_0_55', 'pd_noise_0_60', 'pd_noise_0_65', 'pd_noise_0_70', 'pd_noise_0_75']
    noise_arr = [0.55, 0.60, 0.65, 0.70, 0.75]
    for i in range(len(directory_arr)):
        main(pd_directory=directory_arr[i], noise=noise_arr[i], num_datasets=100, pts_per_dataset=500)
