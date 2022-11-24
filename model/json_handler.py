#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:59:37 2022

@author: sau
"""

import json
import numpy as np
import os
from pathlib import Path
import shutil

with open('images/labelled_cl_images.json','r') as cl_json_file:
    cl_json = json.load(cl_json_file) 
    
filename = np.zeros(len(cl_json), dtype=object) 

for i in range(0,len(cl_json)):
    filename[i] = cl_json[i]["filename"]

#%%
for i in filename:
    # find CL 200 images
    source1 = Path(__file__).parents[2] / 'dataset' /'dataset1'/'CL' / i
    destination1 = Path(__file__).parents[1] / 'images' / 'cl200'
    shutil.copy2(source1, destination1)
    
    # find BSE 200 images
    source2 = Path(__file__).parents[2] / 'dataset' /'dataset1'/'BSE' / i
    destination2 = Path(__file__).parents[1] / 'images' / 'bse200'
    shutil.copy2(source2, destination2)