# Author: Amish Mishra
# Date: January 19, 2022
# README: Generate the PDs of various classes of datasets using different filtrations

import matplotlib.pyplot as plt
import numpy as np
import tadasets
import cechmate as cm


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

    return perturb(data, noise)


def get_pd(filtration_method, data):
    if filtration_method.lower() == "alpha":
        alpha = cm.Alpha(verbose=False)
        filtration = alpha.build(2 * data)  # Alpha goes by radius instead of diameter
        dgms = alpha.diagrams(filtration, verbose=False)

    return dgms


# Initialize variables
noise_level = 0.001
filtration_func = "Alpha"
k = 2   # maximum homology dimension to output into files
shape_name = "Circle"
num_datasets = 50
pts_per_dataset = 100

X = generate_noisy_data(shape_name, noise_level, pts_per_dataset)
dgm = get_pd(filtration_func, X)


# TODO: output pd to text file in appropriate folder
# filename = str(filtration_func + "/PD_n"+noise_level)
# np.savetxt(filename, dgm[0])

