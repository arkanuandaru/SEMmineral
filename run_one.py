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

# run one start
if __name__ == '__main__':
    
    
    #%% bse segmentation
    img_path = "images/bse/image7_20_1"
    img = cv2.imread(img_path + ".tif")
    
    # run  and save segmented image
    img_segmented, _ = segment(img)
    cv2.imwrite(img_path + "seg.tif", img_segmented)

    # show before and after
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))

    ax[0].imshow(img, cmap='gray', interpolation='nearest')
    ax[0].set_title(img_path + '   Original')
    ax[0].axis('off')
    
    ax[1].imshow(img_segmented, cmap='gray', interpolation='nearest')
    ax[1].set_title('Segmented')
    ax[1].axis('off')
    
 