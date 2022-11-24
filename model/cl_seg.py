# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:08:29 2022
@author: arkanuandaru
"""

import cv2
import os
import sys
import shutil

# Histogram based segmentation
from skimage import io
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float, data, io
from skimage.filters import threshold_multiotsu
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd


def segment(bse):
    """
    
    Parameters
    ----------
    bse : TYPE
        DESCRIPTION.
    Returns
    -------
    all_segments_cleaned : TYPE
        DESCRIPTION.
    rel_areas : TYPE
        DESCRIPTION.
    """
    
    # IMAGE LOADING AND DENOISING
    # Clean the noise using edge preserving filter
    
    float_img = img_as_float(bse)
    sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))
    
    denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                                   patch_size=5, patch_distance=3, multichannel=True)
    
    denoise_img_as_8byte = img_as_ubyte(denoise_img)
    
    # Convert the BSE into grayscale
    denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
    
    #%% multi-otsu threshold
    thresholds = threshold_multiotsu(denoise_img_as_8byte_gray, classes = 4)
    regions = np.digitize(denoise_img_as_8byte_gray, bins=thresholds)
    
    # assign segments
    segm1 = (regions == 0) 
    reg1 = (regions == 1) 
    reg2 = (regions == 2) 
    reg3 = (regions == 3) 
    
    # non_porosity count
    reg_count = [np.count_nonzero(reg1), np.count_nonzero(reg2), np.count_nonzero(reg3)]
    
    # Qtz is the region with most data
    if reg_count[0] > reg_count[1]:
        segm2 = reg1 
        segm3 = reg2 + reg3
    else:
        segm2 = reg2
        segm3 = reg1 + reg3
    
    # assign segments
    all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 
    
    all_segments[segm1] = (0,0,0) # the pore 
    all_segments[segm2] = (255,255,0) 
    all_segments[segm3] = (0,255,0) 

    
    
    #%%    
    # normalize brightness
    # denoise_img_as_8byte_gray_ori = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
    # _, denoise_img_as_8byte_gray = cv2.threshold(denoise_img_as_8byte_gray_ori, 254, 255, cv2.THRESH_TRUNC)
    
    # SEGMENTATION THRESHOLD - hardcode
    # segm1 = (denoise_img_as_8byte_gray <= 60) # Porosity
    # segm2 = (denoise_img_as_8byte_gray > 100) & (denoise_img_as_8byte_gray <= 175) # Quartz
    # segm3 = ((denoise_img_as_8byte_gray > 60) & (denoise_img_as_8byte_gray <= 100))\
    #         + (denoise_img_as_8byte_gray > 175) # Other mineral
    # # & (denoise_img_as_8byte_gray > 175)
    # # segm4 = (denoise_img_as_8byte_gray > 210)
    
    #%% Constructing new image shape
    
    #nothing but denoise img size but blank

    all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 
    all_segments[segm1] = (0,0,0) # the pore 2
    all_segments[segm2] = (255,255,0) # quartz 0
    all_segments[segm3] = (0,255,0) # other mineral 3
    # all_segments[segm4] = (255,0,0) # qz overgrowth 1
    
    
    # CLEANING UP
    
    segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
    segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

    segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
    segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

    segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
    segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

    # segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
    # segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))

    all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but 714, 901, 3

    all_segments_cleaned[segm1_closed] = (0,0,0)
    all_segments_cleaned[segm2_closed] = (255,255,0)
    all_segments_cleaned[segm3_closed] = (0,255,0)
    # all_segments_cleaned[segm4_closed] = (1,1,0)
    
    
    # CALCULATING AREA
    # Getting image area
    
    total_area  = all_segments_cleaned.shape[0] * all_segments_cleaned.shape[1]
    pore_area   = len(all_segments_cleaned[segm1_closed]) / total_area
    qz_area     = len(all_segments_cleaned[segm2_closed]) / total_area
    other_area  = len(all_segments_cleaned[segm3_closed]) / total_area
    
    rel_areas = [qz_area, pore_area, other_area]
    
    
    return np.uint8(all_segments_cleaned), rel_areas
    
