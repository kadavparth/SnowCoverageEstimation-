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

