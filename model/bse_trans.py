# -*- coding: utf-8 -*-

import cv2
import numpy as np

def bse_trans(img_name, dataset_path, x, y):
    
    # bse
    bse_seg_path =  dataset_path + "/BSE_segmented/" + img_name
    bse_seg = cv2.imread(bse_seg_path + "_seg.tif")

    # cl
    cl_path =  dataset_path + "/CL/" + img_name
    cl = cv2.imread(cl_path + ".tif")
    
    M = np.float32([[1, 0, x],[0, 1, y]])
    bse_trans = cv2.warpAffine(bse_seg, M, (bse_seg.shape[1], bse_seg.shape[0]))

    # overlay images
    added_image = cv2.addWeighted(cl,0.4,bse_trans,0.1, 0)

    return added_image