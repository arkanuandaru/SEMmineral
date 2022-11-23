"""
Sample submission for the 2022 SPE Europe GeoHackathon, https://www.spehackathon-eu.com.

The code below is intended to demonstrate the input/output requirements for the hackathon submissions.
All code submissions are required to be licensed under an open license, preferably the MIT license as in the present code.

Copyright 2022 Nikolai Andrianov, nia@geus.dk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os, sys
import shutil
import numpy as np
import cv2
from matplotlib import pyplot

def segment(cl):
    """
    A sample procedure to segment a CL image - REPLACE WITH YOUR CODE.

    The code below splits a CL image into the following 4 categories: quartz, quartz overgrowths, other minerals, and
    pores using simple thresholding.

    Returns the segmented CL image and the relative areas of the 4 categories. The color convention in the CL image is
    as follows:
        * Quartz (ID=0): yellow
        * Quartz overgrowths (ID=1): red
        * Pore space (ID=2): black
        * Other minerals (ID=3): green
    """

    # Convert the CL into grayscale
    cl_gray = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY)

    # Get the image area
    total_area = cl_gray.shape[0] * cl_gray.shape[1]

    # Quartz (ID=0): yellow, quartz overgrowths (ID=1): red, pore space (ID=2): black, other minerals (ID=3): green
    color = [(255, 255, 0), (255, 0, 0), (0, 0, 0), (0, 255, 0), (255, 255, 255)]

    # The order reflects roughly how light do these categories appear in the images
    ids = [1, 2, 3, 0]

    # Fix the grayscale thresholds between the respective categories
    thresholds = [100, 145, 150]

    # Start with a segmented image containing just overgrowths (ID=1)
    cl_seg = np.zeros((cl.shape), np.uint8)
    cl_seg[cl_gray >= 0] = color[1][::-1]

    # Keep the classes id's at pixel locations
    cl_seg_id = np.zeros((cl_gray.shape), np.uint8)
    cl_seg_id[cl_gray >= 0] = 1

    # Classify the pixels using corresponding threshold
    for id, thresh in zip(ids[1:], thresholds):
        cl_seg[cl_gray > thresh] = color[id][::-1]
        cl_seg_id[cl_gray >= thresh] = id

    # cv2.imshow('CL_seg', cl_seg)
    # cv2.waitKey(0)

    # Get the relative areas of the categories
    ind, areas = np.unique(cl_seg_id, return_counts=True)
    rel_areas = [area/total_area for area in areas]

    # Make sure that the order of areas corresponds to the ascending order of categories
    rel_areas = [rel_areas for _, rel_areas in sorted(zip(ind, rel_areas))]

    return cl_seg, rel_areas

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
        # Check if the subfolder with CL images exists for the current dataset
        cl_folder = os.path.join(folder, dataset, 'CL')
        if not os.path.isdir(cl_folder):
            print(cl_folder + ' not found..')
            continue

        # Create the folder for segmented CL images
        cl_seg_folder = os.path.join(folder, dataset, 'CL_segmented')
        try:
            if os.path.isdir(cl_seg_folder):
                print('\nDeleting everything in ' + str(cl_seg_folder) + '\n')
                shutil.rmtree(cl_seg_folder)
            os.mkdir(cl_seg_folder)
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (cl_seg_folder, e))
            continue

        # Write the header to the results file
        resfile = os.path.join(folder, dataset, 'results_' + dataset + '.csv')
        try:
            with open(resfile, 'w') as csvfile:
                csvfile.write('path,quartz_rel_area,overgrowth_rel_area,pores_rel_area,otherminerals_rel_area\n')
        except Exception as e:
            print('Failed to create %s. Reason: %s' % (resfile, e))
            continue

        # Loop through all CL files and segment them
        for root, dirs, files in os.walk(cl_folder):

            for file in files:
                # Read the current CL image using OpenCV
                try:
                    cl = cv2.imread(os.path.join(cl_folder, file))
                except Exception as e:
                    print('Failed to open %s. Reason: %s' % (os.path.join(cl_folder, file), e))
                    continue

                # Segment the CL image
                cl_seg, rel_areas = segment(cl)

                # Saving the segmented CL image
                try:
                    cv2.imwrite(os.path.join(cl_seg_folder, file), cl_seg)
                except Exception as e:
                    print('Failed to save %s. Reason: %s' % (os.path.join(cl_seg_folder, file), e))
                    continue

                # Keep the earlier saved results
                lines = []
                if os.path.isfile(resfile):
                    with open(resfile, 'rt') as csvfile:
                        lines = csvfile.readlines()

                # Update the results with the current segmented CL image
                result = os.path.join(cl_folder, file) + ',' + ','.join(str(rel_area) for rel_area in rel_areas) + '\n'
                with open(resfile, 'wt') as csvfile:
                    lines.append(result)
                    for l in lines:
                        csvfile.write(l)