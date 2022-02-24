# Author: Amish Mishra
# Date: February 23, 2022
# README: This file helps to visualize feature importance for a given ML model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use("seaborn")


home = os.path.expanduser("~")
basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/"
expected_features = 11

filtration_func_arr = ["Alpha", "Del_Rips", "Rips"]
directory = 'pd_noise_0_20'
noise = 0.25

i = 1
for f in filtration_func_arr:
    # Loading the iris plants dataset (classification)
    print("Loading", f'{basefilepath}{directory}/{f}/{f}_res_25_df.pkl')
    data = pd.read_pickle(f'{basefilepath}{directory}/{f}/{f}_res_25_df.pkl')
    
    if data.shape[1] != expected_features+1:
        print("Dataframe does not have expected number of columns! Check",
                f'{basefilepath}{directory}/{f}/{f}_res_25_df.pkl')

    # dividing the datasets into two parts i.e. training datasets and test datasets
    X = data.iloc[:,1:]
    y = data.iloc[:,0]

    # creating and training a classifier
    clf = RandomForestClassifier()
    # clf = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # calculate feature importance and rotate array by 90 to have heatmap correspond to PI grid
    feat_imp_H2 = np.rot90(clf.feature_importances_[:25].reshape(5,5))
    feat_imp_H1 = np.rot90(clf.feature_importances_[25:50].reshape(5,5))
    feat_imp_H0 = np.rot90(clf.feature_importances_[50:].reshape(1,5))

    plt.subplot(3,3,i)
    heat_map = sns.heatmap(feat_imp_H0, linewidth = 1 , annot = True)
    plt.title(str(f)+" HeatMap of H0 Image")
    plt.subplot(3,3,i+1)
    heat_map = sns.heatmap(feat_imp_H1, linewidth = 1 , annot = True)
    plt.title(str(f)+" HeatMap of H1 Image")
    plt.subplot(3,3,i+2)
    heat_map = sns.heatmap(feat_imp_H2, linewidth = 1 , annot = True)
    plt.title(str(f)+" HeatMap of H2 Image")
    i+=3
plt.show()