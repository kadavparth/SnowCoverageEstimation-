from lib2to3.pgen2.token import RPAR
import os,sys
from webbrowser import get
import pandas as pd 
import cv2 
import json 
import numpy as np
import time 
from sklearn.preprocessing import normalize 
import json 
import csv

#------------------------------------------------------------
# Preprocessing and Image transformation functions
#------------------------------------------------------------

def read_json(filename):
    '''
    Description
    ----------
    Read parameters from json and return dict.

    Parameters
    ----------
    filename  :  str
        Name of json file to extract values from.

    Return Type  -->  dict
    '''
    PARAM = {}
    try:
        with open(filename, "r") as f:
            data = json.loads(f.read())
            for key, value in data.items():
                PARAM[key] = value
    except FileNotFoundError:
        #print("{} --> DIDN'T FIND {}!".format('FileNotFoundError', filename))
        PARAM = None
    return PARAM

def adjust_with_max(pt, max_x, max_y): 
    if pt[0] > max_x:
        new_x = max_x
    else:
        new_x = pt[0]
    
    if pt[1] > max_y:
        new_y = max_y
    else:
        new_y = pt[1]

    return (new_x, new_y)

def draw_rois(img, ROI_PARAMS):
    
    lane_mask = np.zeros((720,1280),dtype='uint8')
    road_mask = np.zeros((720,1280),dtype='uint8')
    
    h = 720
    w = 1280

    # ##################### LANE ROI ####################
    # Calculate points based on ROI_PARAMS
    lane_bl = adjust_with_max((ROI_PARAMS["lane_roi"]["roi_center_x"] - 0.5*ROI_PARAMS["lane_roi"]["bottom_width"], ROI_PARAMS["lane_roi"]["bottom_y"]), w, h)
    lane_tl = adjust_with_max((ROI_PARAMS["lane_roi"]["roi_center_x"] - 0.5*ROI_PARAMS["lane_roi"]["top_width"], ROI_PARAMS["lane_roi"]["top_y"]), w, h)
    lane_br = adjust_with_max((ROI_PARAMS["lane_roi"]["roi_center_x"] + 0.5*ROI_PARAMS["lane_roi"]["bottom_width"], ROI_PARAMS["lane_roi"]["bottom_y"]), w, h)
    lane_tr = adjust_with_max((ROI_PARAMS["lane_roi"]["roi_center_x"] + 0.5*ROI_PARAMS["lane_roi"]["top_width"], ROI_PARAMS["lane_roi"]["top_y"]), w, h)

    # filter points for max x and y

    lane_pts = np.array((lane_bl, lane_tl, lane_tr, lane_br), np.int32)
    lane_mask = cv2.fillPoly(lane_mask, [lane_pts], color=(255,255,255))

    # UNCOMMENT BELOW LINE TO SHOW LANE ROI
    # out_img = cv2.polylines(out_img, [lane_pts], True, (0, 0, 255), 2)
    
    # ##################### ROAD ROI ####################
    # Calculate points based on ROI_PARAMS
    road_bl = adjust_with_max((ROI_PARAMS["road_roi"]["roi_center_x"] - 0.5*ROI_PARAMS["road_roi"]["bottom_width"] - ROI_PARAMS["road_roi"]["diff"], ROI_PARAMS["road_roi"]["bottom_y"]), w, h)
    road_tl = adjust_with_max((ROI_PARAMS["road_roi"]["roi_center_x"] - 0.5*ROI_PARAMS["road_roi"]["top_width"] + ROI_PARAMS["road_roi"]["diff"], ROI_PARAMS["road_roi"]["top_y"]), w, h)
    road_br = adjust_with_max((ROI_PARAMS["road_roi"]["roi_center_x"] + 0.5*ROI_PARAMS["road_roi"]["bottom_width"] - ROI_PARAMS["road_roi"]["diff"], ROI_PARAMS["road_roi"]["bottom_y"]), w, h)
    road_tr = adjust_with_max((ROI_PARAMS["road_roi"]["roi_center_x"] + 0.5*ROI_PARAMS["road_roi"]["top_width"] + ROI_PARAMS["road_roi"]["diff"], ROI_PARAMS["road_roi"]["top_y"]), w, h)

    # filter points for max x and y

    road_pts = np.array((road_bl, road_tl, road_tr, road_br), np.int32)
    road_mask = cv2.fillPoly(road_mask, [road_pts], color=(255,255,255))
    road_mask = cv2.resize(road_mask, (256,256))
    # out_img = cv2.polylines(out_img, [road_pts], True, (255, 0, 0), 2)
    return lane_mask, road_mask

def get_roi_pixels(raw_image_resize,z):
    v = cv2.bitwise_and(raw_image_resize, raw_image_resize, mask=z)
    pixel_locs = np.argwhere(z == 255)
    roi_pixels = np.ndarray((len(pixel_locs), 3))
    for i, loc in enumerate(pixel_locs):
        roi_pixels[i] = raw_image_resize[loc[0], loc[1]]
    # reshape the image to a 2D array of pixels and 1 color values (Gray)
    # Reshaping to 1D vector
    pixel_values = roi_pixels.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    return pixel_values, pixel_locs

def get_avg_ch_vals(img, mask, values="bgr"):
    # get location of mask pixels
    pixel_locs = np.argwhere(mask == 255)
    b_vals = []
    g_vals = []
    r_vals = []
    for pixel in pixel_locs:
        cube = img[pixel[0], pixel[1]]
        b_vals.append(cube[0])
        g_vals.append(cube[1])
        r_vals.append(cube[2])

    return np.average(b_vals), np.average(g_vals), np.average(r_vals)

def get_std_ch_vals(img, mask, values = "bgr"):
    pixel_locs = np.argwhere(mask == 255)
    b_vals = []
    g_vals = []
    r_vals = []
    for pixel in pixel_locs:
        cube = img[pixel[0], pixel[1]]
        b_vals.append(cube[0])
        g_vals.append(cube[1])
        r_vals.append(cube[2])

    return np.std(b_vals), np.average(g_vals), np.average(r_vals)

# -------------------------------------------------------------
# Data preparation and feature extraction functions
# -------------------------------------------------------------
def preprocessImage(image, roi_mask):
    image = cv2.resize(image, (256,256))
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    return masked_image

# def preprocessLabel(label, roi_mask, size):
#     label = cv2.resize(label, size)
#     masked_label = cv2.bitwise_and(label, label, mask=roi_mask)
#     _, masked_threshed_label = cv2.threshold(masked_label, 50, 255, cv2.THRESH_BINARY)
#     return masked_threshed_label

def getFeatureVector(df_val,image,image_256,ROI_PARAMS,features=['rgb-mean', 'rgb-std', 'rgb-mean-std', 'rgb-mean-vis', 'rgb-std-vis', 'rgb-mean-std-vis']):
    feat_list = []
    lane_mask, road_mask = draw_rois(image, ROI_PARAMS)
    road_vals_mean = get_avg_ch_vals(image_256, road_mask)
    road_vals_std = get_std_ch_vals(image_256, road_mask)
    for feature in features:
        if feature == 'rgb-mean':
            feat_list.append([road_vals_mean[2], road_vals_mean[1], road_vals_mean[0]])
        elif feature == 'rgb-std':
            feat_list.append([road_vals_std[2], road_vals_std[1], road_vals_std[0]])
        elif feature == 'rgb-mean-std':
            feat_list.append([road_vals_mean[2], road_vals_mean[1], road_vals_mean[0], road_vals_std[2], road_vals_std[1], road_vals_std[0]])
        elif feature == 'rgb-mean-vis':
            feat_list.append([road_vals_mean[2], road_vals_mean[1], road_vals_mean[0], df_val])
        elif feature == 'rgb-std-vis':
            feat_list.append([road_vals_std[2], road_vals_std[1], road_vals_std[0], df_val])
        elif feature == 'rgb-mean-std-vis':
            feat_list.append([road_vals_mean[2], road_vals_mean[1], road_vals_mean[0], road_vals_std[2], road_vals_std[1], road_vals_std[0], df_val])
    file = open('/home/parth/AIM2/snow_coverage_estimation/1_get_feature_and_label_sets/all.csv', 'a+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(feat_list)
    return feat_list