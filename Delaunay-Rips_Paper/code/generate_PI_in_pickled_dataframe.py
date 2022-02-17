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

from cmath import pi
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

def make_img_square(pimgr, H1_H2_num_pixels):
    # Make PI for PD a square
    birth_len = pimgr.birth_range[1]-pimgr.birth_range[0]
    pers_len = pimgr.pers_range[1]-pimgr.pers_range[0]
    if pers_len < birth_len:
        pimgr.pers_range = (pimgr.pers_range[0], pimgr.pers_range[0] + birth_len)
    else:
        pimgr.birth_range = (pimgr.birth_range[0], pimgr.birth_range[0] + pers_len)
    pimgr.pixel_size = max(birth_len, pers_len)/H1_H2_num_pixels
    return pimgr


def get_fitted_pimgr(noise_folder, H1_H2_num_pixels, H0_num_pixels):
    # Initialize variables
    home = os.path.expanduser("~")
    basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/{noise_folder}/"
    shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]
    filtration_func_arr = ["Alpha", "Del_Rips", "Rips"]
    num_datasets = 100

    # Find the image range for the H_2 class diagrams
    pdgms_H2 = []
    k = 2   # Hom class to extract
    for filtration_func in filtration_func_arr:
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
    for filtration_func in filtration_func_arr:
        for shape_name in shape_name_arr:
            for i in range(num_datasets):
                path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
                pd = np.load(path)
                pdgms_H1.append(pd)

    # Find the image range for the H_0 class diagrams
    pdgms_H0 = []
    k = 0   # Hom class to extract
    for filtration_func in filtration_func_arr:
        for shape_name in shape_name_arr:
            for i in range(num_datasets):
                path = f"{basefilepath}{filtration_func}/{shape_name}/PD_{i}_{k}.npy"
                pd = np.load(path)
                pdgms_H0.append(pd)

    # Set the persistence image parameters
    p_size = 0.0001
    pimgrH2 = PersistenceImager(pixel_size=p_size)
    pimgrH2.fit(pdgms_H2, skew=True)
    pimgrH2.weight_params = {'n': 1.0}
    pimgrH2.kernel_params = {'sigma': [[pimgrH2.pixel_size/1000, 0.0], [0.0, pimgrH2.pixel_size/1000]]}
    pimgrH1 = PersistenceImager(pixel_size=p_size)
    pimgrH1.fit(pdgms_H1, skew=True)
    pimgrH1.weight_params = {'n': 1.0}
    pimgrH1.kernel_params = {'sigma': [[pimgrH1.pixel_size/1000, 0.0], [0.0, pimgrH1.pixel_size/1000]]}
    # Make PIs square shaped
    pimgrH2 = make_img_square(pimgrH2, H1_H2_num_pixels)
    pimgrH1 = make_img_square(pimgrH1, H1_H2_num_pixels)

    # Set up H_0 PI to be a vertical image window centered on H_0 classes
    pimgrH0 = PersistenceImager(pixel_size=0.00001) # set initial pixel_size very small so that imager can find a tight fit
    pimgrH0.fit(pdgms_H0, skew=True)
    pimgrH0.pixel_size = (pimgrH0.pers_range[1]-pimgrH0.pers_range[0])/H0_num_pixels
    pimgrH0.birth_range = (-pimgrH0.pixel_size/2, pimgrH0.pixel_size/2)
    pimgrH0.weight_params = {'n': 1.0}
    pimgrH0.kernel_params = {'sigma': [[pimgrH0.pixel_size/1000, 0.0], [0.0, pimgrH0.pixel_size/1000]]}

    return pimgrH2, pimgrH1, pimgrH0


def main(filtration_func, num_datasets, noise_folder, H2imgr, H1imgr, H0imgr, verbose=True):
    # Initialize variables
    home = os.path.expanduser("~")
    basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/{noise_folder}/"
    shape_name_arr = ["Circle", "Sphere", "Torus", "Random", "Clusters", "Clusters_in_clusters"]

    # Save resolution of images to be used when running rips and del-rips
    H2_resolution = H2imgr.resolution
    H1_resolution = H1imgr.resolution
    H0_resolution = H0imgr.resolution
    print('Resolutions of H_0, H_1, H_2:', H0_resolution, H1_resolution, H2_resolution)

    # Turn H_2, H_1, H_0 into a flattened PI vector for each shape class and dataset
    data_list = len(shape_name_arr)*[None]*num_datasets
    idx = 0
    shape_idx = 0
    for shape_name in shape_name_arr:
        path = f"{basefilepath}{filtration_func}/{shape_name}/"
        for i in range(num_datasets):
            # Make PI of H_2 diagram
            if shape_name.lower() == 'circle':    
                pimg_H2 = np.full(H2imgr.resolution, 0)
            else:
                filename = str("PD_"+str(i)+"_"+str(2))               
                print(f'{path}{filename}') if verbose else ''
                pd = np.load(f'{path}{filename}.npy')
                pimg_H2 = H2imgr.transform(pd, skew=True)

            # Make PI of H_1 diagram
            filename = str("PD_"+str(i)+"_"+str(1))               
            print(f'{path}{filename}') if verbose else ''
            pd = np.load(f'{path}{filename}.npy')
            pimg_H1 = H1imgr.transform(pd, skew=True)
            # plot_PI(pd, H1imgr, pimg_H1)
            
            # Make PI of H_0 diagram
            filename = str("PD_"+str(i)+"_"+str(0))               
            print(f'{path}{filename}') if verbose else ''
            pd = np.load(f'{path}{filename}.npy')
            pimg_H0 = H0imgr.transform(pd, skew=True)

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
                     'pd_noise_0_30', 'pd_noise_0_35','pd_noise_0_40' , 'pd_noise_0_45', 'pd_noise_0_50',
                     'pd_noise_0_55', 'pd_noise_0_60', 'pd_noise_0_65', 'pd_noise_0_70', 'pd_noise_0_75']

    for directory in directory_arr:
        print('Getting imgr objects ready...')
        H2imgr, H1imgr, H0imgr = get_fitted_pimgr(directory, H1_H2_num_pixels=2, H0_num_pixels=3)
        print('Imgr objects ready')
        for f in filtration_func_arr:
            print("Generating pickle file in", directory, "for", f, "...")
            main(f, 100, directory, H2imgr, H1imgr, H0imgr, verbose=False)
