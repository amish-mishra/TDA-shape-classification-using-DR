# Author: Amish Mishra
# Date: September 14, 2021
# README: Graph ripser, alpha (cechmate), and del-rips on a time vs dimension of d-sphere

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
    radius = 1
    noise = 0.1
    pts = 50
    start_dim = 1
    max_dim = 11
    increment = 1
    max_run_time = 6
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
        print(str(n) + " dim")
        # Data
        # Initialize a noisy circle
        data = tadasets.dsphere(n=pts, d=n, r=radius, noise=noise)

        # Delaunay Rips
        if run_del_rips:
            tic = time.time()
            filtration = DR.build_filtration(data, n)
            dgms_dr = cm.phat_diagrams(filtration, show_inf=True)
            del_rips_time = time.time() - tic
            if del_rips_time < max_run_time:
                del_rips_time_array[i] = del_rips_time
            else:
                run_del_rips = False


        # Ripser
        if run_rips:
            tic = time.time()
            dgms_rips = ripser(data, maxdim=n)['dgms']
            rips_time = time.time() - tic
            if rips_time < max_run_time:
                rips_time_array[i] = rips_time
            else:
                run_rips = False

        # Alpha
        if run_alpha:
            tic = time.time()
            alpha = cm.Alpha()
            alpha_filtration = alpha.build(2*data)
            dgms_alpha = alpha.diagrams(alpha_filtration)
            alpha_time = time.time() - tic
            if alpha_time < max_run_time:
                alpha_time_array[i] = alpha_time
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
    plt.title("Run-time vs Dimension of d-sphere of " + str(pts) + " points\n" +
              "radius=" + str(radius) + " noise=" + str(noise))

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()



