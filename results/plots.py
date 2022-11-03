#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 15:08:53 2022

@author: parth
"""

import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np 


features = ['Red-Mean', 'Green-Mean', 'Blue-Mean', 'Red-Std', 'Green-Std', 'Blue-Std', 'Visibility']

os.chdir('/home/parth/AIM2/snow_coverage_estimation/results/')

file = pd.read_csv('results.csv')
file = file.drop('Unnamed: 0',axis=1)

# test = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/featureset-5_test.npy')
# train = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/featureset-5_train.npy')

# label_test = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/labels_test.npy')
# label_train = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/labels_train.npy')

test = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/featureset-2_test.npy')
train = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/featureset-2_train.npy')

label_test = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/labels_test.npy')
label_train = np.load('/home/parth/AIM2/snow_coverage_estimation/feature_and_label_sets/256x256/labels_train.npy')


X = np.concatenate((test, train))
y = np.concatenate((label_test, label_train))

dtrees = tree.DecisionTreeClassifier()
dtrees.fit(train, label_train)
dtrees.score(test, label_test)

y_test_pred = dtrees.predict(test)

label_test_df = pd.DataFrame(label_test, columns=['label'])
label_test_df = label_test_df.replace({0: 'none', 1: 'standard', 2:'heavy'})
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['label'])
y_test_pred_df = y_test_pred_df.replace({0: 'none', 1: 'standard', 2:'heavy'})

# sort = dtrees.feature_importances_

fig = plt.figure(figsize=(20,5))

ax1 = fig.add_subplot(121)
skplt.estimators.plot_feature_importances(dtrees, feature_names = features, 
                                          title = 'Feature Importance' ,ax=ax1)

#skplt.estimators.plot_feature_importances(dtrees, feature_names= features, ax=ax1)

# indices = np.argsort(sort)

# new_ind = np.array(["Blue-Mean", ""])

# plt.title("Feature Importance")
# plt.barh(range(X.shape[1]), sort[indices], color='r', align='center')
# plt.yticks(range(X.shape[1]), indices)
# plt.ylim([0, X.shape[1]])
# plt.show()
ax2 = fig.add_subplot(122)
skplt.metrics.plot_confusion_matrix(label_test_df, y_test_pred_df,
                                    normalize=True,
                                    title = 'Confusion Matrix',
                                    cmap = 'Oranges')

# a = [1,2,3,4,5,6]

# b = filter(lambda x: x%2 == 0,a)

# print(list(b))
