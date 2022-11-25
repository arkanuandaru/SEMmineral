# -*- coding: utf-8 -*-


import cv2

def save_bse_seg(bse_seg, img_path, img_name):
    bse_seg_rgb = cv2.cvtColor(bse_seg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path + "/BSE_segmented/" + img_name + "_seg.tif", bse_seg_rgb)
    
def save_cl_seg(cl_seg, img_path, img_name):
    cl_seg_rgb = cv2.cvtColor(cl_seg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path + "/CL_segmented/" + img_name + "_seg.tif", cl_seg_rgb)
    