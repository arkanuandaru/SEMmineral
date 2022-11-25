# -*- coding: utf-8 -*-

import os
import cv2

def save_bse_seg(bse_seg, img_path, img_name):
    bse_seg_rgb = cv2.cvtColor(bse_seg, cv2.COLOR_BGR2RGB)
    path = os.path.join(img_path, img_name)
    cv2.imwrite(path + "_seg.tif", bse_seg_rgb)
    
def save_cl_seg(cl_seg, img_path, img_name):
    cl_seg_rgb = cv2.cvtColor(cl_seg, cv2.COLOR_BGR2RGB)
    path = os.path.join(img_path, img_name)
    cv2.imwrite(path + "_seg.tif", cl_seg_rgb)
