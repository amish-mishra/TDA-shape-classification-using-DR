'''
Author: Amish Mishra
Date: Feb 3, 2022
README: Generate the PIs of various PDs from.npy files into a single dataframe with structure
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

from tabnanny import verbose
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import os
from persim import PersistenceImager, plot_diagrams
import pandas


def plot_PI(pd, pimgr, pimg):
    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    axs[0].set_title("Original Diagram")
    pimgr.plot_diagram(pd, skew=False, ax=axs[0])
    axs[1].set_title("Birth-Persistence\nCoordinates")
    pimgr.plot_diagram(pd, skew=True, ax=axs[1])
    axs[2].set_title("Persistence Image")
    pimgr.plot_image(pimg, ax=axs[2])
    plt.tight_layout()
    plt.show()




def main(filtration_func, num_datasets, noise_folder, verbose=True):
    # Initialize variables
    home = os.path.expanduser("~")
    basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/{noise_folder}/"
    shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]

    # Find the image range for the H_2 class diagrams
    pdgms_H2 = []
    k = 2   # Hom class to extract
    for shape_name in shape_name_arr:
        if shape_name.lower()=="circle" and k==2:   # there are no H_2 persistence pairs for the Circle
            continue
        for i in range(num_datasets):
            path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
            pd = np.load(path)
            pdgms_H2.append(pd)

    # Find the image range for the H_1 class diagrams
    pdgms_H1 = []
    k = 1   # Hom class to extract
    for shape_name in shape_name_arr:
        for i in range(num_datasets):
            path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
            pd = np.load(path)
            pdgms_H1.append(pd)

    # Find the image range for the H_0 class diagrams
    pdgms_H0 = []
    k = 0   # Hom class to extract
    for shape_name in shape_name_arr:
        for i in range(num_datasets):
            path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
            pd = np.load(path)
            pdgms_H0.append(pd)

    # Set the persistence image parameters
    pixel_size = 1
    pimgrH2 = PersistenceImager(pixel_size=pixel_size)
    pimgrH2.fit(pdgms_H2, skew=True)
    pimgrH2.weight_params = {'n': 1.0}
    pimgrH2.kernel_params = {'sigma': [[pimgrH2.pixel_size/1000, 0.0], [0.0, pimgrH2.pixel_size/1000]]}
    pimgrH1 = PersistenceImager(pixel_size=pixel_size)
    pimgrH1.fit(pdgms_H1, skew=True)
    pimgrH1.weight_params = {'n': 1.0}
    pimgrH1.kernel_params = {'sigma': [[pimgrH1.pixel_size/1000, 0.0], [0.0, pimgrH1.pixel_size/1000]]}
    pimgrH0 = PersistenceImager(pixel_size=0.00001)
    pimgrH0.fit(pdgms_H0, skew=True)
    pimgrH0.pixel_size = (pimgrH0.pers_range[1]-pimgrH0.pers_range[0])/1
    pimgrH0.birth_range = (-pimgrH0.pixel_size/2, pimgrH0.pixel_size/2)
    pimgrH0.weight_params = {'n': 1.0}
    pimgrH0.kernel_params = {'sigma': [[pimgrH0.pixel_size/1000, 0.0], [0.0, pimgrH0.pixel_size/1000]]}

    # Save resolution of images to be used when running rips and del-rips
    H2_resolution = pimgrH2.resolution
    H1_resolution = pimgrH1.resolution
    H0_resolution = pimgrH0.resolution
    # print('Resolutions of H_0, H_1, H_2:', H0_resolution, H1_resolution, H2_resolution)

    # Turn H_2, H_1, H_0 into a flattened PI vector for each shape class and dataset
    data_list = len(shape_name_arr)*[None]*num_datasets
    idx = 0
    shape_idx = 0
    for shape_name in shape_name_arr:
        path = f"{basefilepath}{filtration_func}/{shape_name}/"
        for i in range(num_datasets):
            # Make PI of H_2 diagram
            if shape_name.lower() == 'circle':    
                pimg_H2 = np.full(pimgrH2.resolution, 0)
            else:
                filename = str("PD_"+str(i)+"_"+str(2))               
                print(f'{path}{filename}') if verbose else ''
                pd = np.load(f'{path}{filename}.npy')
                pimg_H2 = pimgrH2.transform(pd, skew=True)

            # Make PI of H_1 diagram
            filename = str("PD_"+str(i)+"_"+str(1))               
            print(f'{path}{filename}') if verbose else ''
            pd = np.load(f'{path}{filename}.npy')
            pimg_H1 = pimgrH1.transform(pd, skew=True)
            # plot_PI(pd, pimgrH1, pimg_H1)
            
            # Make PI of H_0 diagram
            filename = str("PD_"+str(i)+"_"+str(0))               
            print(f'{path}{filename}') if verbose else ''
            pd = np.load(f'{path}{filename}.npy')
            pimg_H0 = pimgrH0.transform(pd, skew=True)

            # Add vector as a row to data_list
            data_list[idx] = np.concatenate(([shape_idx], pimg_H2.flatten(), pimg_H1.flatten(), pimg_H0.flatten()))
            idx += 1
        shape_idx += 1

    # Make the dataframe for all of the PI data    
    df = pandas.DataFrame(data_list)
    df.rename(columns = {0:'shape_class'}, inplace = True)
    df = df.astype({'shape_class': int})
    print(df) if verbose else ''
    # df.to_pickle(f'{basefilepath}{filtration_func}/{filtration_func}_df.pkl')


if __name__ == '__main__':
    filtration_func_arr = ["Alpha", "Del_Rips", "Rips"]
    directory_arr = ['pd_noise_0_05', 'pd_noise_0_10', 'pd_noise_0_15', 'pd_noise_0_20', 'pd_noise_0_25',
                     'pd_noise_0_30', 'pd_noise_0_35','pd_noise_0_40' , 'pd_noise_0_45', 'pd_noise_0_50']
    for directory in directory_arr:
        for f in filtration_func_arr:
            print("Generating pickle file in", directory, "for", f, "...")
            main(filtration_func=f, num_datasets=100, noise_folder=directory, verbose=False)

exit()
# TODO: work on applying same resolution to rips and del_rips methods
print(alpha_H1_resolution)
pdgms_H1 = []
k = 1   # Hom class to extract
filtration_func = "Rips"
for shape_name in shape_name_arr:
    for i in range(num_datasets):
        path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
        pd = np.load(path)
        pdgms_H1.append(pd)

pimgrH1 = PersistenceImager(pixel_size=0.05)
pimgrH1.resolution = alpha_H1_resolution
pimgrH1.fit(pdgms_H1, skew=True)
pimgrH1.weight_params = {'n': 1.0}
pimgrH1.kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
pimg_H1 = pimgrH1.transform(pdgms_H1[0], skew=True)
print(pimgrH1.resolution)
plot_PI(pdgms_H1[0], pimgrH1, pimg_H1)
