from argparse import ArgumentParser
import os, sys
import numpy as np
import cv2
from sklearn import svm, metrics
import pickle
import time

sys.path.append(os.path.abspath("."))
import tools as TT

# -------------------------------------------------------------
# Define Global Variables (variables used throughout the code)
# -------------------------------------------------------------
CWD = os.getcwd()                   # Current directory path
FEATURE_AND_LABEL_SETS_PATH = CWD+"/feature_and_label_sets/"
TRAINED_MODELS_PATH = CWD+"/trained_models/"

# -------------------------------------------------------------
# Function Definition
# -------------------------------------------------------------
def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("feature_set", type=str, help="Options: 0, 1, 2, 3 -- see 'tire-track-identification/1_get_feature_and_label_sets/feature_sets.json' for description")
    parser.add_argument("model", type=str, help="Options: svm, dt, rf")
    parser.add_argument("--size", type=str, default="75x75", help="image size to find in feature_and_label_sets_folder: example 75x75")

    return parser.parse_args()

# -------------------------------------------------------------
# Reading in training feature vectors
# -------------------------------------------------------------
arguments = parse_arguments()

# Get training vector paths
X_test_path = FEATURE_AND_LABEL_SETS_PATH+arguments.size+"/featureset-{}_test.npy".format(arguments.feature_set)
y_test_path = FEATURE_AND_LABEL_SETS_PATH+arguments.size+"/labels_test.npy"

# Read in
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# -------------------------------------------------------------
# Get desired model, reading from pickled file
# -------------------------------------------------------------

model_name = arguments.model
name = "{}-featureset-{}-{}.sav".format(model_name, arguments.feature_set, arguments.size)

model_file_path = TRAINED_MODELS_PATH+name
model = pickle.load(open(model_file_path, 'rb'))

# -------------------------------------------------------------
# Testing the model
# -------------------------------------------------------------
print("\33[46m\33[30mMaking Predictions...\33[0m\n")
# Make prediction & time it
start_t = time.time()
y_pred = model.predict(X_test)
predict_t = time.time()-start_t
fps = 300.0/predict_t            # 300 frames in test

# Model Timing: how fast the model runs
print("\33[35mFPS:\33[0m",fps)

# Model Accuracy: how often is the classifier correct?
acc = metrics.accuracy_score(y_test.ravel(), y_pred)
print("\33[33mAccuracy:\33[0m",acc)

# Model Precision: what percentage of positive tuples are labeled as such?
precision = metrics.precision_score(y_test.ravel(), y_pred)
print("\33[32mPrecision:\33[0m",precision)

# Model Recall: what percentage of positive tuples are labelled as such?
recall = metrics.recall_score(y_test.ravel(), y_pred)
print("\33[34mRecall:\33[0m",recall)

# Model IoU: what percentage of positive tuples are labelled as such?
jac = metrics.jaccard_score(y_test.ravel(), y_pred)
print("\33[33mIoU:\33[0m",jac)

# Model F1 Score
F1_score = (2*precision*recall)/(precision+recall)
print("\33[32mF1 Score:\33[0m",F1_score)

# Confusion Matrix
cm = metrics.confusion_matrix(y_test.ravel(), y_pred) 
print("\33[34mConfusion Matrix:\33[0m",cm)
TP = cm[0][0]
FP = cm[1][0]
TN = cm[1][1]
FN = cm[0][1]


print("\33[46m\33[30mDONE!\33[0m\n")