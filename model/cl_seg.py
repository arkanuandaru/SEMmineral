# -*- coding: utf-8 -*-

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


def cl_segment(img, cl_segm4):
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
    float_img = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))
    denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                                        patch_size=5, patch_distance=3, multichannel=True)
    
    denoise_img_as_8byte = img_as_ubyte(denoise_img)
    denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
    
    # multi-otsu threshold
    thresholds = threshold_multiotsu(denoise_img_as_8byte_gray, classes = 4)
    regions = np.digitize(denoise_img_as_8byte_gray, bins=thresholds)

    # assign segments
    segm1 = (regions == 0) 
    segm2 = (regions == 1) 
    segm3 = (regions == 2) + (regions == 3) 
    segm4 = cl_segm4
    
    # assign segments
    all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 
    
    all_segments[segm1] = (0,0,0) # the pore 
    all_segments[segm2] = (255,0,0) # Qz overgrowth
    all_segments[segm3] = (255,255,0) # Qz 
    all_segments[segm4] = (0,255,0) # Other
    
    # CLEANING UP
    
    segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
    segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))
    
    segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
    segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))
    
    segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
    segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))
    
    segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
    segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))
    
    all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but 714, 901, 3
    
    all_segments_cleaned[segm1_closed] = (0,0,0)
    all_segments_cleaned[segm2_closed] = (255,0,0)
    all_segments_cleaned[segm3_closed] = (255,255,0)
    all_segments_cleaned[segm4_closed] = (0,255,0)
    
    img_segmented = np.uint8(all_segments_cleaned)

    # CALCULATING AREA
    # Getting image area
    
    total_area  = all_segments_cleaned.shape[0] * all_segments_cleaned.shape[1]
    pore_area   = len(all_segments_cleaned[segm1_closed]) / total_area
    qzove_area  = len(all_segments_cleaned[segm2_closed]) / total_area
    
    rel_areas = [pore_area, qzove_area]
    
    
    return img_segmented, rel_areas
    
