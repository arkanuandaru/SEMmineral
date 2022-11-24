#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:12:59 2022

@author: sau
"""
# standard modules
import sys
import cv2
from matplotlib import pyplot as plt

# custom modules
sys.path.append('/Users/sau/Documents/Stanford/adventure/semhackathon2022/semhackathon/model/')
from bse_seg import *
from bse_trans import *

#%% run one start
if __name__ == '__main__':
   
    # bse segmentation
    img_name = "image6_18_1"
    img_path = "dataset1/trainingset"
    img = cv2.imread(img_path + "/BSE/" + img_name + ".tif")
    
    # run  and save segmented image
    img_segmented, _ = segment(img)
    cv2.imwrite(img_path + "/BSE_segmented/" + img_name + "_seg.tif", img_segmented)
    
    #%% cl bse overlay
    img_overlay = bse_trans(img_name, img_path, 28, 20)
    cv2.imwrite(img_path + "/CL_segmented/" + img_name + "_seg.tif", img_overlay)
    
    # show before and after
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))

    ax[0].imshow(img, cmap='gray', interpolation='nearest')
    ax[0].set_title(img_path + '   Original')
    ax[0].axis('off')
    
    ax[1].imshow(img_overlay, interpolation='nearest')
    ax[1].set_title('BSE_segmented + overlay')
    ax[1].axis('off')
    
    
    

    