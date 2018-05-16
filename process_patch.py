import numpy as np
import openslide
from PIL import Image
from openslide import OpenSlideError
import math
import os

def convert_to_npy(filename):
    img = Image.open(filename) 
    data = np.array(img, dtype = 'uint8')
    return data

def get_white_trans_fracs(patch):
    white = np.mean(patch, 2) > 200
    transparent = np.mean(patch, 2) == 0
    num_pixels = patch.shape[0] * patch.shape[1]
    frac_white = np.sum(white) / num_pixels
    frac_trans = np.sum(transparent) / num_pixels
    return frac_white, frac_trans




### REFACTOR THIS PART
filename = 'data/patient_096_node_2.tif'
#dest_path = 'data/debug_patches/'

good_path = 'data/good_patches/'
bad_path = 'data/bad_patches/'

patch_h = 500
patch_w = 500

slide = openslide.open_slide(filename)

[w, h] = slide.dimensions
#x_TL = 0
#y_TL = 0 # top left corner

x_TL = math.floor(w / 2) # start around the center
y_TL = math.floor(h / 2)

idx = 0 # patch index (start from 0)

x_list = np.arange(start = x_TL, stop = w, step = patch_w) # drop the boarder patch
y_list = np.arange(start = y_TL, stop = h, step = patch_h)

for x in x_list:
    for y in y_list:
        patch = slide.read_region((x,y), 0, (patch_w,patch_h)) # extract patch
        patch_np = np.array(patch, dtype = 'uint8')
        frac_white, frac_trans = get_white_trans_fracs(patch_np)
        
        if frac_white < 0.95 and frac_trans < 0.1:
            patch_name = good_path + str(idx) + '_' + str(x) + '_' + str(y) + '.png'
            patch.save(patch_name)
        else:
            patch_name = bad_path + str(idx) + '_' + str(x) + '_' + str(y) + '.png'
            patch.save(patch_name)
        idx += 1