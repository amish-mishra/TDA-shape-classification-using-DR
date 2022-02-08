# Author: Amish Mishra
# Date: February 7, 2022
# README: This file is to practice creating a random forest classifier


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np


# Loading the iris plants dataset (classification)
iris = datasets.load_iris()
print(iris.target_names)
print(iris.feature_names)

# dividing the datasets into two parts i.e. training datasets and test datasets
X, y = datasets.load_iris( return_X_y = True)
X.tolist()
X[4][0] = [1,2]
print(X[4][0])
exit()

for i in range(len(X)):
    X[i][0]= np.random.rand(1,5)


# Spliting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# importing random forest classifier from assemble module
# creating dataframe of IRIS dataset
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
					'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
					'species': iris.target})

# printing the top 5 datasets in iris dataset
print(data.head())

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

