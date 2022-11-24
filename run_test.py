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

#%% test threshold - setup

img_path = "images/image5_60_3"
img = cv2.imread(img_path + ".tif")
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 

#%test image manipulation
ret, th = cv2.threshold(img, 150, 200, cv2.THRESH_TRUNC)
# plt.imshow(th, cmap=plt.cm.gray, interpolation='nearest')

# show plot
# img_cvt= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(img.flat, bins=100, range=(0,255),label=("ori"))
plt.hist(th.flat, bins=100, range=(0,255), label=("th"))
plt.legend()


cv2.imwrite(img_path + "th.tif", th)

#%% test multi-otsu

from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu
import numpy as np

image = cv2.imread("images/image31_44_2.tif")
float_img = img_as_float(image)
sigma_est = np.mean(estimate_sigma(float_img, channel_axis=1))

denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                               patch_size=5, patch_distance=3, channel_axis=1)

denoise_img_as_8byte = img_as_ubyte(denoise_img)
denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
 
plt.imshow(denoise_img_as_8byte_gray, cmap=plt.cm.gray, interpolation='nearest') 

thresholds = threshold_multiotsu(denoise_img_as_8byte_gray, classes = 4)
regions = np.digitize(denoise_img_as_8byte_gray, bins=thresholds)

# test segmentation

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

img_segmented = np.uint8(all_segments_cleaned)


#Let us look at the input image, thresholds on thehistogram and final segmented image
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(denoise_img_as_8byte , cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(image.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(img_segmented, cmap='gray')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.show()