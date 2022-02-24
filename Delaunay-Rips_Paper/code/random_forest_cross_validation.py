# Author: Amish Mishra
# Date: February 16, 2022
# README: This file is creating a classifier and assesses accuracy using cross validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


home = os.path.expanduser("~")
basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/"
expected_features = 11

filtration_func_arr = ["Alpha", "Del_Rips", "Rips"]
directory_arr = ['pd_noise_0_05', 'pd_noise_0_10', 'pd_noise_0_15', 'pd_noise_0_20', 'pd_noise_0_25',
                 'pd_noise_0_30', 'pd_noise_0_35','pd_noise_0_40' , 'pd_noise_0_45', 'pd_noise_0_50',
                 'pd_noise_0_55', 'pd_noise_0_60', 'pd_noise_0_65', 'pd_noise_0_70', 'pd_noise_0_75']
noise_arr = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

alpha_accuracy = np.empty((len(directory_arr), 1))
del_rips_accuracy = np.empty((len(directory_arr), 1))
rips_accuracy = np.empty((len(directory_arr), 1))
for f in filtration_func_arr:
    i = 0
    for directory in directory_arr:
        # Loading the iris plants dataset (classification)
        print("Loading", f'{basefilepath}{directory}/{f}/{f}_df.pkl')
        data = pd.read_pickle(f'{basefilepath}{directory}/{f}/{f}_df.pkl')
        
        if data.shape[1] != expected_features+1:
            print("Dataframe does not have expected number of columns! Check",
                  f'{basefilepath}{directory}/{f}/{f}_df.pkl')

        # dividing the datasets into two parts i.e. training datasets and test datasets
        X = data.iloc[:,1:]
        y = data.iloc[:,0]

        # creating a classifier
        # clf = RandomForestClassifier()
        clf = svm.SVC(kernel='linear')

        # Cross validated
        cv_results = cross_validate(clf, X, y, cv=10)

        # using metrics module for accuracy calculation
        if f.lower() == 'alpha':
            alpha_accuracy[i] = cv_results['test_score'].mean()
        elif f.lower() == 'del_rips':
            del_rips_accuracy[i] = cv_results['test_score'].mean()
        if f.lower() == 'rips':
            rips_accuracy[i] = cv_results['test_score'].mean()
        i += 1

# Plotting both the curves simultaneously
# print(bott_dist_arr_A, bott_dist_arr_R, bott_dist_arr_DR)
plt.plot(noise_arr, alpha_accuracy, color='g', label='Alpha')
plt.plot(noise_arr, del_rips_accuracy, color='b', label='Del-Rips')
plt.plot(noise_arr, rips_accuracy, color='r', label='Rips')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Noise", fontsize=16)
plt.ylabel("Model Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title("Classifying 6 shape classes using Random Forest:\n"+
#           str(1-test_size)+"-"+str(test_size)+" training-test split", fontsize=16)
plt.grid(b=True, axis='y')

# Adding legend, which helps us recognize the curve according to it's color
plt.legend(fontsize=14)

plt.show()