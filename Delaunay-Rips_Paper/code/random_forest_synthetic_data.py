# Author: Amish Mishra
# Date: February 7, 2022
# README: This file is creating a random forest classifier and assess accuracy


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import os


home = os.path.expanduser("~")
basefilepath = f"{home}/Documents/research/Delaunay-Rips_Paper/pd_noise_0_05/"

# Loading the iris plants dataset (classification)
data = pd.read_pickle(f'{basefilepath}alpha_df.pkl') 

print(data)
# dividing the datasets into two parts i.e. training datasets and test datasets
X = data.iloc[:,1:]
y = data.iloc[:,0]

# Spliting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100, min_samples_leaf = 5)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

if data.shape[1]-1 == 3: # only 3 pixels used to make PIs
    data.rename(columns = {1: 'H_2_PI', 2: 'H_1_PI', 3: 'H_0_PI'}, inplace = True)
    feature_imp = pd.Series(clf.feature_importances_, index = ['H_2_PI', 'H_1_PI', 'H_0_PI']).sort_values(ascending = False)
    print(feature_imp)

