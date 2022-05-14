# Author: Amish Mishra
# Date: September 15, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a time vs number of points graph for 2-sphere with averaging

import os
from ripser import ripser
import cechmate as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tadasets
import time
home = os.path.expanduser("~")
basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/"
import sys
# appending the directory of Del_Rips.py
# in the sys.path list
sys.path.append(f'{basefilepath}code')
import Del_Rips as DR

from persim import plot_diagrams
# matplotlib.rcParams['font.family'] = 'helvetica'


if __name__ == '__main__':
    # Initialize variables
    trials = 1
    radius = 1
    noise = 0.1
    start_pts = 50
    max_pts = 550   #1550
    increment = 50  #100
    fixed_dim = 1   #2
    max_run_time = 7
    rips_time_array = [None] * (int((max_pts - start_pts) / increment))
    alpha_time_array = [None] * (int((max_pts - start_pts) / increment))
    del_rips_time_array = [None] * (int((max_pts - start_pts) / increment))
    rips_time = 0
    del_rips_time = 0
    alpha_time = 0

    # Boolean variable to cap the run time of each method
    run_del_rips = True
    run_rips = True
    run_alpha = True

    i = 0
    for n in range(start_pts, max_pts, increment):
        del_rips_trial_time_array = [None]*trials
        rips_trial_time_array = [None]*trials
        alpha_trial_time_array = [None]*trials
        for t in range(trials):
            # Data
            print(str(n) + " pts")
            data = tadasets.dsphere(n=n, d=fixed_dim, r=radius, noise=noise)

            # Delaunay Rips
            if run_del_rips:
                tic = time.time()
                filtration = DR.build_filtration(data, fixed_dim)
                dgms_dr = cm.phat_diagrams(filtration, show_inf=True)
                del_rips_time = time.time() - tic
                del_rips_trial_time_array[t] = del_rips_time

            # Ripser
            if run_rips:
                tic = time.time()
                dgms_rips = ripser(data, maxdim=fixed_dim)['dgms']
                rips_time = time.time() - tic
                rips_trial_time_array[t] = rips_time

            # Alpha
            if run_alpha:
                tic = time.time()
                alpha = cm.Alpha()
                alpha_filtration = alpha.build(2*data)
                dgms_alpha = alpha.diagrams(alpha_filtration)
                alpha_time = time.time() - tic
                alpha_trial_time_array[t] = alpha_time
        
        if run_del_rips:
            avg_del_rips_time = np.median(del_rips_trial_time_array)
            if avg_del_rips_time < max_run_time:
                del_rips_time_array[i] = avg_del_rips_time
            else:
                run_del_rips = False
        if run_rips:
            avg_rips_time = np.median(rips_trial_time_array)
            if avg_rips_time < max_run_time:
                rips_time_array[i] = avg_rips_time
            else:
                run_rips = False
        if run_alpha:
            avg_alpha_time = np.median(alpha_trial_time_array)
            if avg_alpha_time < max_run_time:
                alpha_time_array[i] = avg_alpha_time
            else:
                run_alpha = False        

        i += 1

    X = np.arange(start_pts, max_pts, increment)

    # Plotting both the curves simultaneously
    plt.plot(X, rips_time_array, color='r', label='Rips', linewidth=5, solid_capstyle='round')
    plt.plot(X, del_rips_time_array, color='b', label='Del-Rips', linewidth=5, solid_capstyle='round')
    plt.plot(X, alpha_time_array, color='g', label='Alpha', linewidth=5, solid_capstyle='round')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Number of points", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    # plt.title("Run-time vs Number of points\n Noisy " + str(fixed_dim) + "-sphere" +
    #           ", radius=" + str(radius) + ", noise=" + str(noise)+ "\n(Plot of medians of "+str(trials)+" trials)", fontsize=16)

    plt.xticks(fontsize=14, ticks=np.linspace(start_pts, max_pts, 5))
    plt.yticks(fontsize=14)#, ticks=np.linspace(min(np.concatenate([rips_time_array, del_rips_time_array, alpha_time_array])),
                           #                   max(np.concatenate([rips_time_array, del_rips_time_array, alpha_time_array])), 5))

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend(fontsize=14)

    # To load the display window
    plt.show()



