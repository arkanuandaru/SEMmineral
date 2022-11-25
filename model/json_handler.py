#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
from pathlib import Path
import shutil
import pandas as pd

with open('dataset1/labelled_cl_images.json','r') as cl_json_file:
    cl_json = json.load(cl_json_file) 

#%% get training dataset    
filename = np.zeros(len(cl_json), dtype=object) 

for i in range(0,len(cl_json)):
    filename[i] = cl_json[i]["filename"]


for i in filename:
    # find CL 200 images
    source1 = Path(__file__).parents[2] / 'dataset' /'dataset1'/'CL' / i
    destination1 = Path(__file__).parents[1] / 'images' / 'cl200'
    shutil.copy2(source1, destination1)
    
    # find BSE 200 images
    source2 = Path(__file__).parents[2] / 'dataset' /'dataset1'/'BSE' / i
    destination2 = Path(__file__).parents[1] / 'images' / 'bse200'
    shutil.copy2(source2, destination2)
    
#%% construct df for training
df_json = pd.DataFrame()

for i in range(0, len(cl_json)):
    
    filename_30 = pd.Series(index=range(30), dtype=object)
    
    for j in range(0,30):
        filename_30[j] = cl_json[i]["filename"]
        
    label_id = pd.Series(cl_json[i]["label_id"])
    label_x = pd.Series(cl_json[i]["label_x"])
    label_y = pd.Series(cl_json[i]["label_y"])
    
    df_json_each = pd.concat([filename_30, label_id, label_x, label_y],axis=1,keys =["filename","label_id","label_x","label_y"],ignore_index=True)
    df_json = pd.concat([df_json, df_json_each],ignore_index=True)
