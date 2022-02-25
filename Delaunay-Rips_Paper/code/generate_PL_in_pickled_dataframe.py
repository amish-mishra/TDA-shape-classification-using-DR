'''
Author: Amish Mishra
Date: Feb 24, 2022
README: Generate the Persistence Landscapes of various PDs from.npy files into a single dataframe with structure
Class | H_2 pixel 1 | H_2 pixel 2 |....| H_1 pixel 1 | H_1 pixel 2 | ..... H_0 pixel 1 .....
0     |             |             |    |             |             |
0     |             |             |    |             |             |
0     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
1     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
2     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
.     |             |             |    |             |             |
5     |             |             |    |             |             |
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import persim.landscapes
import pandas


def main(filtration_func, num_datasets, noise_folder, verbose=True, output_file=False):
    # Initialize variables
    home = os.path.expanduser("~")
    basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/{noise_folder}/"
    shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]
    data_list = len(shape_name_arr)*[None]*num_datasets
    idx = 0
    shape_idx = 0

    for shape_name in shape_name_arr:
        path = f"{basefilepath}{filtration_func}/{shape_name}/"
        for i in range(num_datasets):
            pds = []
            for k in range(3):  # reconstruct PDs of each dataset
                if shape_name.lower() == 'circle' and k == 2:    
                    continue
                else:
                    filename = str("PD_"+str(i)+"_"+str(k))               
                    print(f'{path}{filename}') if verbose else ''
                    pd = np.load(f'{path}{filename}.npy')
                    pds.append(pd)
            # TODO: Unable to compute and plot Persistence landscape
            pl = persim.landscapes.PersLandscapeExact(pds, hom_deg=1)
            print(pl)
            persim.landscapes.plot_landscape_simple(pl, title=f"{filtration_func}: {shape_name}")
            plt.show()
            # Add vector as a row to data_list
            # data_list[idx] = np.concatenate(([shape_idx]))
            idx += 1
        shape_idx += 1

    return 
    # Make the dataframe for all of the PI data    
    df = pandas.DataFrame(data_list)
    df.rename(columns = {0:'shape_class'}, inplace = True)
    df = df.astype({'shape_class': int})
    print(df) if verbose else ''
    if output_file:
        df.to_pickle(f'{basefilepath}{filtration_func}/{filtration_func}_res_25_df.pkl')
    else:
        print("WARNING: No pickled df is being saved")


if __name__ == '__main__':
    filtration_func_arr = ["Alpha", "Del_Rips", "Rips"]
    directory_arr = ['pd_noise_0_05', 'pd_noise_0_10', 'pd_noise_0_15', 'pd_noise_0_20', 'pd_noise_0_25',
                     'pd_noise_0_30', 'pd_noise_0_35', 'pd_noise_0_40', 'pd_noise_0_45', 'pd_noise_0_50',
                     'pd_noise_0_55', 'pd_noise_0_60', 'pd_noise_0_65', 'pd_noise_0_70', 'pd_noise_0_75']
    
    filtration_func = "Alpha"
    num_datasets = 1
    noise_folder = 'pd_noise_0_05'
    main(filtration_func, num_datasets, noise_folder, verbose=True, output_file=False)




