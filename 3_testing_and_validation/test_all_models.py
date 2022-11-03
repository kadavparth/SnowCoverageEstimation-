from argparse import ArgumentParser
from genericpath import isfile
import os, sys
import numpy as np
from numpy.lib import column_stack
import pandas as pd
import cv2
from sklearn import svm, metrics
import pickle
import time
import glob

sys.path.append(os.path.abspath("."))
import tools as TT

# -------------------------------------------------------------
# Define Global Variables (variables used throughout the code)
# -------------------------------------------------------------
CWD = os.getcwd()                   # Current directory path
FEATURE_AND_LABEL_SETS_PATH = CWD+"/feature_and_label_sets/"
TRAINED_MODELS_PATH = CWD+"/trained_models/"
RESULTS_PATH = CWD+"/results/"

# -------------------------------------------------------------
# Function Definition
# -------------------------------------------------------------
model_paths = glob.glob(TRAINED_MODELS_PATH+"*")
results_dict = {"ML_method": [],
                "Feature_set": [],
                "Size": [],
                "Accuracy": [],
                "FPS": [],
                "Precision": [],
                "Recall": [],
                "IoU": [],
                "F1_Score": []
                }
for model_path in model_paths:
    # -------------------------------------------------------------
    # Reading in training feature vectors
    # -------------------------------------------------------------
    file_name = model_path.split("/")[-1]
    model_name = file_name.split("-")[0]
    n_set = file_name.split("-")[2]
    size = file_name.split("-")[3][:-4]    
    
    # Get training vector paths
    X_test_path = FEATURE_AND_LABEL_SETS_PATH+size+"/featureset-{}_test.npy".format(n_set)
    y_test_path = FEATURE_AND_LABEL_SETS_PATH+size+"/labels_test.npy"

    # Read in
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # -------------------------------------------------------------
    # Get desired model, reading from pickled file
    # -------------------------------------------------------------
    model = pickle.load(open(model_path, 'rb'))

    # # -------------------------------------------------------------
    # # Testing the model
    # # -------------------------------------------------------------
    print("\33[46m\33[30mMaking Predictions with {}... \33[0m\n".format(file_name))
    # Make prediction & time it
    start_t = time.time()
    y_pred = model.predict(X_test)
    predict_t = time.time()-start_t
    avg_predict_time = len(y_test)/predict_t

    # # y_pred1 = model.predict([X_test[1]])
    # sum = 0
    # count = 0 
    # fps = []
    
    # i = 0
    
    # while i < 6265:
    #     to = time.time()
    #     y_pred = model.predict([X_test[i]])
    #     end_time = time.time()
    #     sum += end_time
    #     if sum > 1.0:
    #         fps.append(count)
    #     else:
    #         count += 1
    #     i += 1

    # fps = np.average(fps)

    # Model Timing: how fast the model runs
    print("\33[35mAverage Predict Time:\33[0m",(avg_predict_time))

    # Model Accuracy: how often is the classifier correct?
    acc = metrics.accuracy_score(y_test.ravel(), y_pred)
    print("\33[33mAccuracy:\33[0m",acc)

    # Model Precision: what percentage of positive tuples are labeled as such?
    precision = metrics.precision_score(y_test.ravel(), y_pred,average='micro')
    print("\33[32mPrecision:\33[0m",precision)

    # Model Recall: what percentage of positive tuples are labelled as such?
    recall = metrics.recall_score(y_test.ravel(), y_pred,average='micro')
    print("\33[34mRecall:\33[0m",recall)

    # Model IoU: what percentage of positive tuples are labelled as such?
    jac = metrics.jaccard_score(y_test.ravel(), y_pred,average='micro')
    print("\33[33mIoU:\33[0m",jac)

    # Model F1 Score
    F1_score = (2*precision*recall)/(precision+recall)
    print("\33[32mF1 Score:\33[0m",F1_score)

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_test.ravel(), y_pred) 
    print("\33[34mConfusion Matrix:\33[0m",cm)

    # Saving data to dictionary
    results_dict["ML_method"].append(model_name)
    results_dict["Feature_set"].append(n_set)
    results_dict["Size"].append(size)
    results_dict["Accuracy"].append(acc)
    results_dict["Precision"].append(precision)
    results_dict["Recall"].append(recall)
    results_dict["FPS"].append((avg_predict_time))
    results_dict["IoU"].append(jac)
    results_dict["F1_Score"].append(F1_score)

    print("\33[46m\33[30mDONE!\33[0m\n")

results_df = pd.DataFrame(results_dict, columns=results_dict.keys())

try:
    if os.path.exists(RESULTS_PATH):
        results_df.to_csv(RESULTS_PATH+"results.csv")
    else:
        os.makedirs(RESULTS_PATH)
except:
    pass
