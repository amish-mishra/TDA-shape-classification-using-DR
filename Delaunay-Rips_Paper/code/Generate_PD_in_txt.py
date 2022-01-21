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

def generate_noisy_data(shape, noise, pts):
    def perturb(data, noise):       # local function that perturbs the data by the noise level
        perturb_vects = np.random.randn(len(data), len(data[0]))
        mags = np.linalg.norm(perturb_vects, axis=1)
        for i, mag in enumerate(mags):
            perturb_vects[i] /= mag * (
                        1 / noise)  # check that this is actually appying to the whole row. Also, perturb by at MOST noise, not exactly noise
        perturbed_data = data + perturb_vects
        return perturbed_data

    if shape.lower() == "circle":
        data = tadasets.dsphere(n=pts, d=1, r=1, noise=0)
    elif shape.lower() == "sphere":
        data = tadasets.dsphere(n=pts, d=2, r=1, noise=0)
    elif shape.lower() == "torus":
        data = tadasets.torus(n=pts, c=1, a=0.5, noise=0)
    return perturb(data, noise)


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


# TODO: add functionality for more shape types
# Initialize variables
noise_level = 0.01
filtration_func_arr = ["Alpha", "Rips", "Del_Rips"]
k = 2   # maximum homology dimension to output into files
shape_name_arr = ["Circle", "Sphere", "Torus"]
num_datasets = 25
pts_per_dataset = 100

for i in range(num_datasets):
    for shape_name in shape_name_arr:
        X = generate_noisy_data(shape_name, noise_level, pts_per_dataset)   # generate data for a shape
        for filtration_func in filtration_func_arr: # compute PDs using each filtration for a fixed dataset
            dgm = get_pd(filtration_func, X)
            for j in range(k+1):  # put H_0, H_1,... classes in separate text files
                if j < len(dgm):
                    home = os.path.expanduser("~")
                    filepath = f"{home}/Documents/Del_Rips_Paper/research/Delaunay-Rips_Paper/pd_noise_0_05/" \
                               f"{filtration_func}/{shape_name}/"
                    filename = str("PD_"+str(i)+"_"+str(j))
                    np.savetxt(f"{filepath}{filename}.txt", dgm[j])
                    print("Finished making", f"{filepath}{filename}.txt")

