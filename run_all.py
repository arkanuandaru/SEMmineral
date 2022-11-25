#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# standard modules
import sys
import cv2
import os
import pathlib
from matplotlib import pyplot as plt
from tqdm import tqdm

# custom modules
sys.path.append(os.getcwd() + '/model/')
from bse_seg import *
from cl_seg import *
from bse_trans import *
from save_seg import *
from rel_calc import *

if __name__ == '__main__':

    # access dataset
    
    dataset_names = ["dataset1_run1", "dataset1_run2", "dataset1_run3","dataset2", "dataset3", "dataset4"]
    folder = os.getcwd()
    datasets = [f for f in dataset_names if os.path.isdir(os.path.join(folder, f))]
    
    # Loop through the datasets
    for dataset in datasets:
        # Check if the subfolder with BSE images exists for the current dataset
        bse_folder = os.path.join(folder, dataset, 'BSE')
        
        # print start confirmation for each dataset
        print("******************************")
        print(f"{dataset} segmentation is starting...\n")
                
        if not os.path.isdir(bse_folder):
            print(bse_folder + ' not found..')
            continue
        
        # Create the folder for segmented BSE images
        bse_seg_folder = os.path.join(folder, dataset, 'BSE_segmented')
       
        try:
            if os.path.isdir(bse_seg_folder):
                # print('\nDeleting everything in ' + str(bse_seg_folder) + '\n')
                shutil.rmtree(bse_seg_folder)
            os.mkdir(bse_seg_folder)
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (bse_seg_folder, e))
            continue
        
        cl_seg_folder = os.path.join(folder, dataset, 'CL_segmented')   
        try:
            if os.path.isdir(cl_seg_folder):
                # print('\nDeleting everything in ' + str(cl_seg_folder) + '\n')
                shutil.rmtree(cl_seg_folder)
            os.mkdir(cl_seg_folder)
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (cl_seg_folder, e))
            continue
        
        cl_trn_folder = os.path.join(folder, dataset, 'CL_trained')   
        try:
            if not os.path.isdir(cl_trn_folder):
                # print('\nDeleting everything in ' + str(cl_seg_folder) + '\n')
                os.mkdir(cl_trn_folder)
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (cl_trn_folder, e))
            continue
        
        # Write the header to the results file
        resfile = os.path.join(folder, dataset, 'results_' + dataset + '.csv')
        try:
            with open(resfile, 'w') as csvfile:
                csvfile.write('path, qz_rel_area, qzovergrowth_rel_area, pore_rel_area,othermin_rel_area\n')
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (resfile, e))
            continue
        
        # ask user for overlay shift
        print("Please enter overlay shifting values. Right and down is positive, left and up is negative.\n If value is unknown, please enter 0 for both.\n")
        shift_x = int(input(f"Enter overlay x_shift for {dataset}: "))
        shift_y = int(input(f"Enter overlay y_shift {dataset}: "))
        
        print(f"\n{dataset} segmentation in progress...")
        
       # Loop through all BSE files and segment them
        for root, dirs, files in os.walk(bse_folder):
            
            for file in files:
                filename = file[:-4]
                
                # Read the current BSE image using OpenCV
                file_extension = pathlib.Path(file).suffix
                if file_extension == ".tif":
                    try:
                        bse = cv2.imread(os.path.join(bse_folder, file))
                    except Exception as e:
                        print('Failed to open %s. Reason: %s' % (os.path.join(bse_folder, file), e))
                        continue
                
                    # Segment the BSE image and save
                    img_bse_seg, cl_segm4 = bse_segment(bse)
                    save_bse_seg(img_bse_seg, bse_seg_folder, filename)
                
                    # assign overlay shifting
                    img_overlay, cl_segm4_trans = bse_trans(filename, dataset, cl_segm4, shift_x, shift_y)
                    
                    # elif file[-5] == 2: 
                    # if file[-5] == 1: 
                    #     img_overlay, cl_segm4_trans = bse_trans(filename, dataset, cl_segm4, 28, 20)
                    # elif file[-5] == 2: 
                    #     img_overlay, cl_segm4_trans = bse_trans(filename, dataset, cl_segm4, 0, 0)
                    # elif file[-5] == 3: 
                    #     img_overlay, cl_segm4_trans = bse_trans(filename, dataset, cl_segm4, 11, 103)
                    # else: 
                    #     img_overlay, cl_segm4_trans = bse_trans(filename, dataset, cl_segm4, 0, 0)
                        
                    # cl segmentation 
                    img_cl_seg, _ = cl_segment(img_overlay, cl_segm4_trans)
                    save_cl_seg(img_cl_seg, cl_seg_folder, filename)
                    
                    # // TODO apply optimizer for cl segmentation using trained ML model
                    save_cl_seg(img_cl_seg, cl_trn_folder, filename)
                    #
                    #
                
                    # calculate areas from optimizer
                    rel_areas = rel_calc(filename, cl_trn_folder, shift_x, shift_y) 
                    
                    # Keep the earlier saved results
                    lines = []
                    if os.path.isfile(resfile):
                        with open(resfile, 'rt') as csvfile:
                            lines = csvfile.readlines()

                    #%Update the results with the current segmented BSE image
                    result = os.path.join(bse_folder, file) + ',' + ','.join(str(rel_area) for rel_area in rel_areas) + '\n'
                    with open(resfile, 'wt') as csvfile:
                        lines.append(result)
                        for l in lines:
                            csvfile.write(l)
                            
            # print end confirmation for each dataset
            print(f"\n{dataset} segmentation is done!")
            print("******************************")