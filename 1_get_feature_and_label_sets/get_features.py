from locale import D_FMT
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import json
import numpy as np
import time
from sklearn import svm
from sklearn import metrics
from alive_progress import alive_it

sys.path.append(os.path.abspath("."))
import tools as TT


# -------------------------------------------------------------
# Define Global Variables (variables used throughout the code)
# -------------------------------------------------------------
RESIZE = (256, 256)                   # resized image size used for ML
CWD = os.getcwd()                   # Current directory path
ROI_MASK_ID = 1                     # 0 - Lane ROI, 1 - Road ROI 
SC_WS_PATH = CWD+"/workspace/" # path to workspace relative to this folder
FEATURE_AND_LABEL_SETS_PATH = CWD + "/feature_and_label_sets/"
# DO NOT CHANGE IMG_SIZE!!!! Use RESIZE to adjust ML training image size
IMG_SIZE = (720, 1280)              # DO NOT CHANGE THIS

# -------------------------------------------------------------
# Prepare folder for saving feature sets and label sets
# -------------------------------------------------------------
if not os.path.isdir(FEATURE_AND_LABEL_SETS_PATH):
    os.mkdir(FEATURE_AND_LABEL_SETS_PATH)

subdir = "{}x{}/".format(RESIZE[0], RESIZE[1])
if not os.path.isdir(FEATURE_AND_LABEL_SETS_PATH+subdir):
    os.mkdir(FEATURE_AND_LABEL_SETS_PATH+subdir)


# -------------------------------------------------------------
# Read in dataset and feature sets
# -------------------------------------------------------------
# Read in dataset and split into train/test
sc_dataset = pd.read_csv(SC_WS_PATH+ "image_label_vis1_ptype.csv")
feature_sets = TT.read_json(CWD+"/1_get_feature_and_label_sets/feature_sets.json")

# Get masks for roi
roi_params = TT.read_json(SC_WS_PATH+"rois.json")

# 1) Train/Test Split
train_df, test_df = train_test_split(sc_dataset, test_size = 0.3, shuffle=True)

# Printing image size to alert user
print("\n\33[6m\33[43m\33[30mIMAGE SIZE = {}x{}\33[0m\n".format(RESIZE[0], RESIZE[1]))

for n_set,feature_set in alive_it(feature_sets.items()):
    X_train = []; y_train = []; X_test = []; y_test = []

    print("\33[36m\33[40mGetting train set for feature set\33[0m", n_set)
    try: 
        for i,row in train_df.iterrows():
            img = cv2.imread(row['IMAGE'])
            img_256 = cv2.imread(row["IMAGE"][0:52] + '256x256_image' + row["IMAGE"][57:])
            # img_n = TT.preprocessImage(img, masks)

            label = row["LABEL"]

            X = TT.getFeatureVector(row['vis1_coeff'],img,img_256,roi_params,features=feature_set)
            y = label
            # print(row["IMAGE"])

            X_train.append(X)
            y_train.append(y)
    except:
        print('error')
    X_train = np.row_stack(X_train)
    y_train = np.row_stack(y_train)

    print("\33[36m\33[40mGetting test set for feature set\33[0m", n_set)
    try:
        for i, row in test_df.iterrows():

            #### FOR DEVELOPMENT PURPOSES ONLY -- REMOVE AFTER CODE IS GOOD!!!! ########
            img = cv2.imread(row["IMAGE"])
            img_256 = cv2.imread(row["IMAGE"][0:52] + '256x256_image' + row["IMAGE"][57:])
            # pixel_locs = np.argwhere(masks == 255)
            # img = TT.preprocessImage(img_n, masks)

            label = row["LABEL"]
            
            # Getting feature vector and label vector
            X = TT.getFeatureVector(row['vis1_coeff'],img,img_256,roi_params, features=feature_set)
            y = label

            X_test.append(X)
            y_test.append(y)
    except:
        print('error')
    X_test = np.row_stack(X_test)
    y_test = np.row_stack(y_test)

    print("\33[32m\33[1mSaving X_test with shape {} and x_train with shape {}\33[0m\n".format(X_test.shape, X_train.shape))
    time.sleep(1)

    np.save(FEATURE_AND_LABEL_SETS_PATH+subdir+"featureset-{}_train.npy".format(n_set),X_train)
    np.save(FEATURE_AND_LABEL_SETS_PATH+subdir+"featureset-{}_test.npy".format(n_set), X_test)

print("\33[32m\33[1mSaving y_test with shape {} and y_train with shape {}\33[0m".format(y_test.shape, y_train.shape))
# only save once
np.save(FEATURE_AND_LABEL_SETS_PATH+subdir+"labels_train.npy", y_train)
np.save(FEATURE_AND_LABEL_SETS_PATH+subdir+"labels_test.npy", y_test)