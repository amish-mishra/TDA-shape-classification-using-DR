# Author: Amish Mishra
# Date: May 10, 2022
# README: Generate a visual of all 6 shape classes for varying noise

from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import tadasets
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


shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]
noise_level = 0.75
pts_per_dataset = 500
fig = plt.figure(figsize=(9,6))

for i, shape_name in enumerate(shape_name_arr):
    X = generate_noisy_data(shape_name, noise_level, pts_per_dataset)   # generate data for a shape
    xs = X[:,0]
    ys = X[:,1]
    if X.shape[1] > 2:  # if points are in 3 dimensions, get the z coordinate
        zs = X[:,2]
    else:   # if points are in 2 dimensions, just put 0's for the z coordinate
        zs = np.zeros(len(X))
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.scatter(xs, ys, zs, marker='.')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    plt.gca().view_init(68, -14)
    ax.set_title(shape_name, fontsize=20)
    ax.grid(False)

plt.show()