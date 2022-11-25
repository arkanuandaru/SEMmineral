# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

def rel_calc(trained_cl_seg, cl_trn_folder, shift_x, shift_y):
    img = cv2.imread(os.path.join(cl_trn_folder,trained_cl_seg) + "_seg.tif")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # crop shifted areas
    if shift_x >= 0 and shift_y >= 0:
        img_crop = img_rgb[shift_x:, shift_y:]
    elif shift_x < 0 and shift_y >= 0:
        img_crop = img_rgb[:shift_x, shift_y:] 
    elif shift_x >= 0 and shift_y < 0:
        img_crop = img_rgb[shift_x:, :shift_y]
    elif shift_x < 0 and shift_y < 0:
        img_crop = img_rgb[:shift_x, :shift_y]  
    
    # calculate segments
    mask_yellow = cv2.inRange(img_crop, (255,255,0), (255,255,0))
    mask_red = cv2.inRange(img_crop, (255,0,0), (255,0,0))
    mask_green = cv2.inRange(img_crop, (0,255,0), (0,255,0))
    mask_black = cv2.inRange(img_crop, (0,0,0), (0,0,0))
    
    # discount black areas due to overlay shifting
    # shifted_area = np.abs((shift_x+1) * (shift_y+1))
    total_area = img_crop.shape[0] * img_crop.shape[1]
    # total_area = mask_black.shape[0] * mask_black.shape[1] - shifted_area

    qz = np.count_nonzero(mask_yellow) / total_area
    qzove = np.count_nonzero(mask_red) / total_area
    other = np.count_nonzero(mask_green) / total_area
    pore = np.count_nonzero(mask_black) / total_area
    # pore = (np.count_nonzero(mask_black) - shifted_area) / total_area

    rel_areas = [qz, qzove, other, pore]
    
    return rel_areas, img_crop