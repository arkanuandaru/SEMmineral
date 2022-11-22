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
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import numpy as np




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
    
    ####################################################################################
    #
    # IMAGE LOADING AND DENOISING
    
    # Clean the noise using edge preserving filter
    float_img = img_as_float(bse)
    
    sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))
    
    denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                                   patch_size=5, patch_distance=3, multichannel=True)
    
    denoise_img_as_8byte = img_as_ubyte(denoise_img)
    
    # Convert the BSE into grayscale
    denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
    
    ####################################################################################
    #
    # SEGMENTATION THRESHOLD

    segm1 = (denoise_img_as_8byte_gray <= 125) # Porosity
    segm2 = (denoise_img_as_8byte_gray > 125) & (denoise_img_as_8byte_gray <= 175) # Quartz
    segm3 = (denoise_img_as_8byte_gray > 175) # Other mineral
    # segm4 = (denoise_img_as_8byte_gray > 210)
    
    ####################################################################################
    #
    # Constructing new image shape
    all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but denoise img size but blank

    all_segments[segm1] = (0,0,0) # the pore 2
    all_segments[segm2] = (255,255,0) # quartz 0
    all_segments[segm3] = (0,255,0) # other mineral 3
    # all_segments[segm4] = (255,0,0) # qz overgrowth 1
    
    
    
    ####################################################################################
    #
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
    
    
    ####################################################################################
    #
    # CALCULATING AREA
    
    # Getting image area
    total_area = all_segments_cleaned.shape[0] * all_segments_cleaned.shape[1]
    
    pore_area = len(all_segments_cleaned[segm1_closed]) / total_area
    qz_area = len(all_segments_cleaned[segm2_closed]) / total_area
    other_area = len(all_segments_cleaned[segm3_closed]) / total_area
    
    rel_areas = [qz_area, pore_area, other_area]
    
    
    return np.uint8(all_segments_cleaned), rel_areas
    

if __name__ == '__main__':
    
    # Get a list of available datasets
    dataset_names = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
    folder = os.getcwd()
    datasets = [f for f in dataset_names if os.path.isdir(os.path.join(folder, f))]
    if not datasets:
        print('No datasets are available! Exiting..')
        sys.exit(1)
        
    
    # Loop through the datasets
    for dataset in datasets:
        # Check if the subfolder with BSE images exists for the current dataset
        bse_folder = os.path.join(folder, dataset, 'BSE')
        if not os.path.isdir(bse_folder):
            print(bse_folder + ' not found..')
            continue
        
        
        # Create the folder for segmented BSE images
        bse_seg_folder = os.path.join(folder, dataset, 'BSE_segmented')
        try:
            if os.path.isdir(bse_seg_folder):
                print('\nDeleting everything in ' + str(bse_seg_folder) + '\n')
                shutil.rmtree(bse_seg_folder)
            os.mkdir(bse_seg_folder)
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (bse_seg_folder, e))
            continue
        
        
        # Write the header to the results file
        resfile = os.path.join(folder, dataset, 'results_' + dataset + '.csv')
        try:
            with open(resfile, 'w') as csvfile:
                csvfile.write('path,quartz_rel_area,pores_rel_area,otherminerals_rel_area\n')
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (resfile, e))
            continue
        
        
        # Loop through all BSE files and segment them
        for root, dirs, files in os.walk(bse_folder):
            
            for file in files:
                # Read the current BSE image using OpenCV
                try:
                    bse = cv2.imread(os.path.join(bse_folder, file))
                except Exception as e:
                    print('Failed to open %s. Reason: %s' % (os.path.join(bse_folder, file), e))
                    continue
                
                # Segment the CL image
                bse_seg, rel_areas = segment(bse)

                # Saving the segmented BSE image
                try:
                    cv2.imwrite(os.path.join(bse_seg_folder, file), bse_seg)
                except Exception as e:
                    print('Failed to save %s. Reason: %s' % (os.path.join(bse_seg_folder, file), e))
                    continue

                # Keep the earlier saved results
                lines = []
                if os.path.isfile(resfile):
                    with open(resfile, 'rt') as csvfile:
                        lines = csvfile.readlines()

                # Update the results with the current segmented BSE image
                result = os.path.join(bse_folder, file) + ',' + ','.join(str(rel_area) for rel_area in rel_areas) + '\n'
                with open(resfile, 'wt') as csvfile:
                    lines.append(result)
                    for l in lines:
                        csvfile.write(l)
    
    
    
    
    
    
    
    


#######################################################################
#
#

img = cv2.imread("dataset1/BSE/image5_58_3.tif")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')  
plt.imshow(img_gray, cmap=plt.cm.gray, interpolation='nearest')  


#######################################################################
#
#Let's clean the noise using edge preserving filter.
#As mentioned in previous tutorial, my favorite is NLM

float_img = img_as_float(img)
sigma_est = np.mean(estimate_sigma(float_img, multichannel=True))

denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                               patch_size=5, patch_distance=3, multichannel=True)
                           
denoise_img_as_8byte = img_as_ubyte(denoise_img)
plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')

denoise_img_as_8byte_gray = cv2.cvtColor(denoise_img_as_8byte, cv2.COLOR_BGR2GRAY)
plt.imshow(denoise_img_as_8byte_gray, cmap=plt.cm.gray, interpolation='nearest')  


########################################################################
#
#Let's look at the histogram to see howmany peaks we have. 
#Then pick the regions for our histogram segmentation.

plt.hist(denoise_img_as_8byte_gray.flat, bins=100, range=(0,255))  #.flat returns the flattened numpy array (1D)

segm1 = (denoise_img_as_8byte_gray <= 100) # Porosity
segm2 = (denoise_img_as_8byte_gray > 100) & (denoise_img_as_8byte_gray <= 175) # Quartz
segm3 = (denoise_img_as_8byte_gray > 175) # Other mineral
# segm4 = (denoise_img_as_8byte_gray > 210)


#How to show all these images in single visualization?
#Construct a new empty image with same shape as original except with 3 layers.
# print(median_img.shape[0])
all_segments = np.zeros((denoise_img_as_8byte_gray.shape[0], denoise_img_as_8byte_gray.shape[1], 3)) #nothing but denoise img size but blank

all_segments[segm1] = (0,0,0) # the pore
all_segments[segm2] = (255,255,0) # quartz
all_segments[segm3] = (0,255,0) # other mineral
# all_segments[segm4] = (1,1,0)
plt.imshow(all_segments)


########################################################################
#
#Lot of yellow dots, red dots and stray dots. how to clean
#We can use binary opening and closing operations. Open takes care of isolated pixels within the window
#Closing takes care of isolated holes within the defined window

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

plt.imshow(all_segments_cleaned)  #All the noise should be cleaned now






img_a = cv2.imread("dataset1/BSE/image5_60_3.tif")
img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
norm_a = np.zeros(img_a.shape)

img_b = cv2.imread("dataset1/BSE/image31_45_2.tif")
img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
