#!/usr/bin/python

# ------------------------------------------------------------------------------
# 
# img_to_csv.py
# 
# Usage:   $ python  img_to_csv.py  /path/to/img.jpg
# 
# Use this script to convert an image to .csv to give as input to Spatial
# 
# ------------------------------------------------------------------------------

import skimage.io
import skimage.transform
import numpy as np
import sys

if len(sys.argv) not in [2, 3, 4, 5]:
  print 'Use this script to convert an image to .csv to give as input to Spatial'
  print 
  print 'Usage:   $ python  img_to_csv.py  /path/to/img.jpg  img_size(default=224,224,3)  img_mean_RGB(default=None)  img_scale(default=None)'
  print 
  print 'Example: $ python  img_to_csv.py  /path/to/img.jpg'
  print 'Example: $ python  img_to_csv.py  /path/to/img.jpg  224,224,3  123.68,116.78,103.94  255.0'
  sys.exit(0)
img_path = sys.argv[1]

dims = [224, 224, 3]
if len(sys.argv) > 2:
  dims_str = sys.argv[2].split(',')
  dims = []
  for dim in dims_str:
    dims.append(int(dim))
assert dims[0] == dims[1]

means = [0., 0., 0.]
if len(sys.argv) > 3:
  means_str = sys.argv[3].split(',')
  means = []
  for mean in means_str:
    means.append(float(mean))

scale = 1
if len(sys.argv) > 4:
  scale_str = sys.argv[4]
  scale = float(scale_str)

# returns image of shape [224, 224, 3] by default
# [height, width, channels]
def load_image(path, size=224):
  img = skimage.io.imread(path)
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
  resized_img = skimage.transform.resize(crop_img, (size, size))
  resized_img[:,:,0] = resized_img[:,:,0]*scale - means[0]
  resized_img[:,:,1] = resized_img[:,:,1]*scale - means[1]
  resized_img[:,:,2] = resized_img[:,:,2]*scale - means[2]
  return resized_img

img = load_image(img_path, dims[0])
img = img.reshape((1, dims[0], dims[1], dims[2]))

# This is in row-major format
np.savetxt('input0.csv', np.transpose(img, [0, 3, 1, 2])[0,:,:,:].flatten())
print 'Input saved to ./input0.csv. You can move this file and update the path in the .scala file for the Spatial DNN'
