# Author: Amish Mishra
# Date: May 15, 2022
# README: Generate a visual of various data shapes

from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import tadasets
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def perturb(data, max_noise):       # function that perturbs the data by the noise level
    perturb_vects = np.random.randn(len(data), len(data[0]))
    mags = np.linalg.norm(perturb_vects, axis=1)
    for i, mag in enumerate(mags):
        noise = np.random.rand()*max_noise # perturb by a length of no more than 'max_noise'
        perturb_vects[i] *= noise/mag
    return data + perturb_vects


def clusters_data(pts, noise):
    # centers = np.random.rand(3, 3);
    centers = [[-.8,.4,0],[.7,.5,0],[.2,-.8,0]]
    dist_matrix = cdist(centers, centers, metric='euclidean')
    centers = centers/dist_matrix.max() # scale centers to have diam 1 here because it is faster to do it now
    a = np.random.randint(np.floor(.1*pts), np.floor(.45*pts))  # random number of points for first cluster, can be at most 45% of all points, but is at least 10%
    b =  np.random.randint(np.floor(.1*(pts-a)), np.floor(.45*(pts-a))) # random number of pts in 2nd cluster, at least 10% remaining, at most 45%
    c = pts-a-b
    data = np.concatenate((np.tile(centers[0,:], (a, 1)), np.tile(centers[1,:], (b, 1)),  np.tile(centers[2,:], (c, 1))))
    return perturb(data, noise)


def x_shape(pts, noise):
    first_half = int(pts/2)
    second_half = pts - first_half
    x1 = np.linspace(-5,5,first_half)
    y1 = np.linspace(-5,5, first_half)
    data1 = np.dstack((x1,y1))[0]
    x2 = np.linspace(-5,5,first_half)
    y2 = -np.linspace(-5,5, first_half)
    data2 = np.dstack((x2,y2))[0]
    data = np.concatenate((data1, data2))
    return perturb(data, noise)




datalst = [None]*7
datalst[0] = tadasets.dsphere(n=100, d=1, noise=0.05)
datalst[1] = clusters_data(100, 0.2)
datalst[2] = np.random.rand(100,2)
datalst[3] = tadasets.swiss_roll(n=100, noise=0.5)
datalst[4] = np.concatenate((tadasets.dsphere(n=50, d=1, noise = 0.05),np.array([0,2])+tadasets.dsphere(n=50, d=1, noise = 0.05)), axis=0)
datalst[5]= x_shape(100, 0.5)
datalst[6] = tadasets.dsphere(n=250, d=2, noise=0.05)


fig = plt.figure(figsize=(14,2))
for i in range(6):
    X = datalst[i]
    ax = fig.add_subplot(1, 7, i+1)
    xs = X[:,0]
    ys = X[:,1]
    ax.scatter(xs, ys, marker='.')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

X = datalst[6]
xs = X[:,0]
ys = X[:,1]
zs = X[:,2]
ax = fig.add_subplot(1, 7, 7, projection='3d')
ax.scatter(xs, ys, zs, marker='.')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
# plt.gca().view_init(68, -14)
# ax.set_axis_off()
ax.grid(False)


plt.show()