from argparse import ArgumentParser
from ast import arguments
from genericpath import exists
import os,sys
from xmlrpc.client import TRANSPORT_ERROR 
import cv2 
import numpy as np 
import pickle 
from alive_progress import alive_it
import time


# Importing all Machine Learning models from sklearn 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sys.path.append(os.path.abspath('.'))
import tools as TT

# --------------------------------------------------------
# defining global variables and using throughout the code
# --------------------------------------------------------

CWD = os.getcwd()
FEATURE_AND_LABEL_SETS_PATH = CWD + '/feature_and_label_sets/'
TRAINED_MODELS_PATH = CWD + "/trained_models/"
feature_sets = TT.read_json(CWD + '/1_get_feature_and_label_sets/feature_sets.json')

#------------------------------------------------------------
#  function definition 
# ----------------------------------------------------------

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, nargs='?', default="rforest", help="Options: svm, dtress, rforest, naive, knn, logistic, linear")
    parser.add_argument("--size", type=str, nargs='?', default="256x256", help='image size to find in the features_and_label_sets folder')
    return parser.parse_args()

# --------------------------------------------------------
#  make folders if not present 
# -------------------------------------------------------

if not os.path.isdir(TRAINED_MODELS_PATH):
    os.mkdir(TRAINED_MODELS_PATH)
else:
    pass
# --------------------------------------------------------
# Reading in training features 
# --------------------------------------------------------

arguments = parse_arguments()

for n_set,feature_set in alive_it(feature_sets.items()):
     
    X_train_path = FEATURE_AND_LABEL_SETS_PATH + arguments.size + "/featureset-{}_train.npy".format(n_set)
    y_train_path = FEATURE_AND_LABEL_SETS_PATH + arguments.size + "/labels_train.npy"

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    # --------------------------------------------------------
    # Train the models after selecting which ones to use 
    # --------------------------------------------------------

    if arguments.model == "svm":
        model_name = arguments.model  
        model = svm.SVC()
        name = '{}-featureset-{}-{}.sav'.format(model_name, n_set, arguments.size)
    elif arguments.model == "dtrees":
        model_name = arguments.model
        model = DecisionTreeClassifier()
        name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)
    elif arguments.model == "rforest":
        model_name = arguments.model
        model = RandomForestClassifier()
        name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)
    elif arguments.model == "naive":
        model_name = arguments.model
        model = GaussianNB()
        name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)
    elif arguments.model == "knn":
        model_name = arguments.model
        model = KNeighborsClassifier()
        name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)
    elif arguments.model == "logistic":
        model_name = arguments.model
        model = LogisticRegression()
        name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)
    # elif arguments.model == "linear":
    #     model_name = arguments.model
    #     model = LinearRegression()
    #     name="{}-featureset-{}-{}.sav".format(model_name, n_set, arguments.size)

    # -------------------------------------------------------------
    # Training & saving the model
    # -------------------------------------------------------------

    print("\n\33[35m\33[5mTraining {} model using feature set {} at resolution {}...\33[0m".format(model_name, n_set, arguments.size))
    model.fit(X_train, y_train.ravel())

    model_file_path = TRAINED_MODELS_PATH + name

    pickle.dump(model, open(model_file_path, 'wb'))
    print("\33[1mModel saved to: \33[33m{}\33[0m\n".format(model_file_path))


print("\33[5m\33[46m\33[30mDONE!\33[0m\n")