# Author: Amish Mishra
# Date: September 14, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a time vs dimension of d-sphere with averaging

import Del_Rips as DR
from ripser import ripser
import cechmate as cm
import numpy as np
import matplotlib.pyplot as plt
import tadasets
import time
from persim import plot_diagrams



if __name__ == '__main__':
    # Initialize variables
    trials = 5
    radius = 1
    noise = 0.1
    pts = 25
    start_dim = 1
    max_dim = 4
    increment = 1
    max_run_time = 3
    rips_time_array = [None] * (int((max_dim - start_dim) / increment))
    alpha_time_array = [None] * (int((max_dim - start_dim) / increment))
    del_rips_time_array = [None] * (int((max_dim - start_dim) / increment))
    rips_time = 0
    del_rips_time = 0
    alpha_time = 0

    # Boolean variable to cap the run time of each method
    run_del_rips = True
    run_rips = True
    run_alpha = True

    i = 0
    for n in range(start_dim, max_dim, increment):
        del_rips_trial_time_array = [None]*trials
        rips_trial_time_array = [None]*trials
        alpha_trial_time_array = [None]*trials
        for t in range(trials):
            print(str(n) + " dim")
            # Data
            data = tadasets.dsphere(n=pts, d=n, r=radius, noise=noise)

            # Delaunay Rips
            if run_del_rips:
                tic = time.time()
                filtration = DR.build_filtration(data, n)
                dgms_dr = cm.phat_diagrams(filtration, show_inf=True)
                del_rips_time = time.time() - tic
                del_rips_trial_time_array[t] = del_rips_time

            # Ripser
            if run_rips:
                tic = time.time()
                dgms_rips = ripser(data, maxdim=n)['dgms']
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
                plt.boxplot(del_rips_trial_time_array, showfliers=False, positions=[n], widths=1)
            else:
                run_del_rips = False
        if run_rips:
            avg_rips_time = np.median(rips_trial_time_array)
            if avg_rips_time < max_run_time:
                rips_time_array[i] = avg_rips_time
                plt.boxplot(rips_trial_time_array, showfliers=False, positions=[n], widths=1)
            else:
                run_rips = False
        if run_alpha:
            avg_alpha_time = np.median(alpha_trial_time_array)
            if avg_alpha_time < max_run_time:
                alpha_time_array[i] = avg_alpha_time
                plt.boxplot(alpha_trial_time_array, showfliers=False, positions=[n], widths=1)
            else:
                run_alpha = False

        i += 1

    X = np.arange(start_dim, max_dim, increment)

    # Plotting both the curves simultaneously
    plt.plot(X, rips_time_array, color='r', label='Rips')
    plt.plot(X, del_rips_time_array, color='b', label='Del-Rips')
    plt.plot(X, alpha_time_array, color='g', label='Alpha')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Dimension")
    plt.ylabel("Time (seconds)")
    plt.title("Run-time vs Dimension\n d-sphere of " + str(pts) + " points" +
              ", radius=" + str(radius) + ", noise=" + str(noise)+"\n(Plot of medians of "+str(trials)+" trials)")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()



