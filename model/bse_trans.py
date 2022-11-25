# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

def bse_trans(img_name, dataset_path, cl_segm4, x, y):
    
    # bse
    bse_seg_path =  os.path.join(dataset_path, "BSE_segmented", img_name)
    bse_seg = cv2.imread(bse_seg_path + "_seg.tif")

    # cl
    cl_path =  os.path.join(dataset_path,"CL", img_name)
    cl = cv2.imread(cl_path + ".tif")
    
    M = np.float32([[1, 0, x],[0, 1, y]])
    bse_trans = cv2.warpAffine(bse_seg, M, (bse_seg.shape[1], bse_seg.shape[0]))
    
    # overlay images
    added_image = cv2.addWeighted(cl,0.4,bse_trans,0.1, 0)
    
    # shift cl_segm4 
    # right & down
    if x > 0 and y > 0:
        cl_segm4_trans = np.pad(cl_segm4,((y,0),(x,0)), mode='constant')[:cl_segm4.shape[0], :cl_segm4.shape[1]]
    # left & down
    elif x < 0 and y > 0:
        cl_segm4_trans = np.pad(cl_segm4,((y,0),(0,x)), mode='constant')[:cl_segm4.shape[0], :cl_segm4.shape[1]] 
    # right & up
    elif x > 0 and y < 0:
        cl_segm4_trans = np.pad(cl_segm4,((0,y),(x,0)), mode='constant')[:cl_segm4.shape[0], :cl_segm4.shape[1]] 
    # left & up
    elif x < 0 and y < 0:
        cl_segm4_trans = np.pad(cl_segm4,((0,y),(0,x)), mode='constant')[:cl_segm4.shape[0], :cl_segm4.shape[1]]
    else: 
        cl_segm4_trans = np.pad(cl_segm4,((y,0),(x,0)), mode='constant')[:cl_segm4.shape[0], :cl_segm4.shape[1]]        
    
    return added_image, cl_segm4_trans