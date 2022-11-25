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

img_path = "dataset1/trainingset/CL/image6_18_1"
img = cv2.imread(img_path + ".tif")
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 

ret, th = cv2.threshold(img, 150, 200, cv2.THRESH_TRUNC)
plt.imshow(th, cmap=plt.cm.gray, interpolation='nearest')

# show plot
# img_cvt= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(img.flat, bins=100, range=(0,255),label=("ori"))
plt.hist(th.flat, bins=100, range=(0,255), label=("th"))
plt.legend()


#%% bse segmentation test denoise

img_path = "images/cl/image31_44_2"
image = cv2.imread(img_path + ".tif")
float_img = img_as_float(image)

# # #%%add blur
# float_blur_img = cv2.GaussianBlur(float_img, (15, 15), 2)
# sigma_est = np.mean(estimate_sigma(float_blur_img, multichannel=True))
# denoise_img = denoise_nl_means(float_blur_img, h=1.3 * sigma_est, fast_mode=False, 
#                                 patch_size=10, patch_distance=6, multichannel=True)                               

# denoise_img_as_8byte = img_as_ubyte(denoise_img)
# denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)

# denoise                                    
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))
denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                                    patch_size=5, patch_distance=3, multichannel=True)



denoise_img_as_8byte = img_as_ubyte(denoise_img)
denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)


# plot before after
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
 
# ax[0].imshow(image)
# ax[0].set_title('Original')
# ax[0].axis('off')

# ax[1].imshow(denoise_img_as_8byte_gray, cmap='gray')
# ax[1].set_title('Denoise')
# ax[1].axis('off')


# test multi-otsu

from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu
import numpy as np

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


cv2.imwrite(img_path + "seg.tif", img_segmented)

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
ax[2].imshow(img_segmented)
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.show()

#%% test bse-cl overlay and translate dataset 1 run 1

# bse
img_path1 = "dataset1/trainingset/BSE_segmented/image6_18_1_seg"
image1 = cv2.imread(img_path1 + ".tif")

# cl
img_path2 = "dataset1/trainingset/CL/image6_18_1"
image2 = cv2.imread(img_path2 + ".tif")

#translate bse
x = 28
y = 20
M_dataset1_1 = np.float32([[1, 0, x],[0, 1, y]])
image1_trans = cv2.warpAffine(image1, M_dataset1_1, (image1.shape[1], image1.shape[0]))

# overlay images
added_image = cv2.addWeighted(image2,0.4,image1_trans,0.1, 0)

plt.imshow(added_image, interpolation='nearest')

# cv2.imwrite('combined.png', added_image)

#%% test cl thresholding

from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu
import numpy as np

img_path = "dataset1/trainingset/CL_segmented/image6_18_1_seg"
img = cv2.imread(img_path + ".tif")
float_img = img_as_float(img)

# denoise                                    
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))
denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                                    patch_size=5, patch_distance=3, multichannel=True)

denoise_img_as_8byte = img_as_ubyte(denoise_img)
denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)

# test threshold
thresholds = threshold_multiotsu(denoise_img_as_8byte_gray, classes = 4)
regions = np.digitize(denoise_img_as_8byte_gray, bins=thresholds)

# test segmentation
# assign segments
segm1 = (regions == 0) 
segm2 = (regions == 1) 
segm3 = (regions == 2) + (regions == 3) 

# assign segments
all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 

all_segments[segm1] = (0,0,0) # the pore 
all_segments[segm2] = (255,0,0) # Qz overgrowth
all_segments[segm3] = (255,255,0) # Qz 


# CLEANING UP

segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but 714, 901, 3

all_segments_cleaned[segm1_closed] = (0,0,0)
all_segments_cleaned[segm2_closed] = (255,0,0)
all_segments_cleaned[segm3_closed] = (255,255,0)

img_segmented = np.uint8(all_segments_cleaned)


# cv2.imwrite(img_path + "seg.tif", img_segmented)

#Let us look at the input image, thresholds on thehistogram and final segmented image
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(denoise_img_as_8byte, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(denoise_img_as_8byte.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(img_segmented)
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.show()