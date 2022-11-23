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

if __name__ == '__main__':
    
    # grab image
    img_path = "images/image5_60_3"
    img = cv2.imread(img_path + ".tif")
    plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')  
    
    # run bse segmentation and save segmented image
    img_segmented, _ = segment(img)
    plt.imshow(img_segmented)
    cv2.imwrite(img_path + "seg.tif", img_segmented)
