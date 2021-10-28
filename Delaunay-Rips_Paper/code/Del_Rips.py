# Author: Amish Mishra
# Date: September 14, 2021
# README: An efficient implementation of Delaunay-Rips

import numpy as np
import matplotlib.pyplot as plt
import tadasets
import time
from scipy.spatial import Delaunay
import itertools


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def find_subsets(s, n):
    return list(itertools.combinations(s, n))


def build_filtration(data, hom_dim):
    # Compute Delaunay 1-skeleton
    tri = Delaunay(data)
    # First add all 0 simplices
    filtration = [([i], 0) for i in range(len(data))]
    dict_for_simplices = {} # track the simplices and their weights to avoid adding duplicates to filtration
    for simplex in tri.simplices:
        simplex = sorted(simplex)
        for dim in range(2, hom_dim+3):
            # assumption: find_subsets will never reverse the order of a subset
            # e.g. the subset (1,2) will never show up as (2,1)
            faces = find_subsets(simplex, dim)
            for face in faces:
                if face not in dict_for_simplices and dim == 2:
                    # assumption: Delaunay triangulation labeled vertices in the same order the data was inputted
                    d = euclidean(data[face[0]], data[face[1]])
                    dict_for_simplices[face] = d
                    filtration.append((list(face), d))
                elif face not in dict_for_simplices and dim > 2:   # simplex needs the weight of the max co-face
                    sub_faces = find_subsets(face, dim-1)
                    max_weight = -1.0
                    for sub_face in sub_faces:
                        weight = dict_for_simplices[sub_face]
                        if weight > max_weight:
                            max_weight = weight
                    dict_for_simplices[face] = max_weight
                    filtration.append((list(face), max_weight))
    return filtration


if __name__ == '__main__':
    n_sphere_dim = 2  # the points will be in n+1 dimensions
    max_hom_dim = n_sphere_dim
    pts = 1000
    data = tadasets.dsphere(n=pts, d=n_sphere_dim, r=1, noise=0.1)
    fig = plt.figure()

    # Delaunay Rips
    tic = time.time()
    filtration = build_filtration(data, max_hom_dim)
    direct_del_rips_time = time.time() - tic
    print("Filtration time:", direct_del_rips_time)
    # dgms = cm.phat_diagrams(filtration, show_inf=True)
    # plot_diagrams(dgms[0])
    # fig.suptitle("Built filtration directly for PHAT\n(Time: %.3gs)" % direct_del_rips_time)

    # Alpha
    # alpha = cm.Alpha()
    # alpha_filtration = alpha.build(2*data)
    # dgms_alpha = alpha.diagrams(alpha_filtration)
    # print("Alpha dgms")
    # print(dgms_alpha)
    # plot_diagrams(dgms_alpha)

    # plt.show()
