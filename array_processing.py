# This module contains the functions to normalize an array for use in convolutional neural networks.
# All functions are called in the function preprocess_array(image, maxima, image_size) which takes a 
# numpy array encoding a multiband image, an array of maximum values for each band, and a target image size.
# It returns a normalized array.


import rasterio
from rasterio.fill import fillnodata
import os
import numpy as np

# Image resize by pixel
def resize(img, image_size):
  '''
  Takes an image as an array and a target image size in pixels. 
  Expects input array of with shape formatted as (bands, rows, columns).
  The function first clips the image to the central 200 rows and columns and 
  then adds a padding of zeros to bring the image to the target size.
  
  '''
  rows = img.shape[1]
  cols = img.shape[2]
  rows_diff = rows - 200
  cols_diff = cols - 200
  r_start = 0
  c_start = 0
  if rows_diff > 0:
    r_start = rows_diff // 2
  if cols_diff > 0:
    c_start = cols_diff // 2
  img = img[:, r_start:r_start+200, c_start:c_start+200]
  
  bands, new_rows, new_cols = img.shape
  if new_rows < image_size[0]:
     new_row_diff = image_size[0] - new_rows
     begin_rows = new_row_diff // 2
     end_rows = new_row_diff - begin_rows
     begin_pad = np.zeros(shape = (bands, begin_rows, new_cols))
     end_pad = np.zeros(shape = (bands, end_rows, new_cols))
     new_image = np.append(begin_pad, img, axis = 1)
     new_image = np.append(new_image, end_pad, axis = 1)
     img = new_image

  if new_cols < image_size[1]:
     new_col_diff = image_size[1] - new_cols
     begin_cols = new_col_diff // 2
     end_cols = new_col_diff - begin_cols
     begin_pad = np.zeros(shape = (bands, image_size[0], begin_cols))
     end_pad = np.zeros(shape = (bands, image_size[0], end_cols))
     new_image = np.append(begin_pad, img, axis = 2)
     new_image = np.append(new_image, end_pad, axis = 2)
     img = new_image
     
  return img

# Fill holes in raster dataset by interpolation from the edges 
def fill_nans(img):
  img_ma = np.ma.MaskedArray(img, np.isnan(img))
  for band in range(img.shape[0]):
     img[band] = fillnodata(img_ma[band], smoothing_iterations = 5, max_search_distance=300.0)
  return img

# Data normalization
def normalize_data(img, maxima):
    for band in range(img.shape[0]):
      img[band][img[band] > maxima[band]] = maxima[band] # This sets all values that are greater than the stored maximum to the maximum
      img[band] = img[band]/maxima[band]
    return img

# Final array processing
def preprocess_array(image, maxima, image_size):
   image = resize(image, image_size)
   image = fill_nans(image)
   image = normalize_data(image, maxima)
   return image