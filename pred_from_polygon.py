# This module contains all of the functions and code necessary to take a polygon from a user as an ee.Geometry 
# object and produce a series of average predictions across polygons inside that area.
# All functions are called in the function PCD_prediction(polygon) which takes a single polygon as an input and
# returns a grid of polygons representing the predicted likelihood of a porphyry copper deposit.

import ee
import geemap
import tensorflow as tf
import json
import pandas as pd
import geopandas as gpd
import math
import numpy as np
import requests
from rasterio.io import MemoryFile
from rasterio.fill import fillnodata

import preprocessing
import band_engineering
import ui_array_processing
from prediction_model import model

# ee.Authenticate()
ee.Initialize()

# Maximum values in all eleven bands of training data
band_maxima = np.array([2.24831426e+04, 1.15938261e+03, 6.38604960e+01, 8.76501817e+01,
       1.68952216e+02, 4.88885296e+02, 5.72398994e+01, 6.85249355e+02,
       9.62179553e+00, 6.53775279e+02, 1.92785899e+02])


# Meters to degrees conversion
def meters_to_degrees_conversion(lat, lon, grid_distance_m):
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412*math.cos(lon) - 93.5*math.cos(3*lon) + 0.118*math.cos(5*lon)

    lat_step_deg = grid_distance_m / m_per_deg_lat
    lon_step_deg = grid_distance_m / m_per_deg_lon

    return (lat_step_deg, lon_step_deg)

# Collecting min and max of longitude and latitude coordinates of a polygon 
def poly_min_max_coords(polygon):
  coordinates = polygon.coordinates().getInfo()
  coordinates_numpy = np.array(coordinates[0])
  lon_min = coordinates_numpy[:,0].min()
  lon_max = coordinates_numpy[:,0].max()
  lat_min = coordinates_numpy[:,1].min()
  lat_max = coordinates_numpy[:,1].max()
  min_max_coordinates = np.array([[lon_max,lat_max],[lon_min,lat_min]])
  return min_max_coordinates

# Create grid points based on the min and max longitude and latitude of the polygon
def create_input_grid_points(min_max_coordinates):
  lat_spacing, lon_spacing = meters_to_degrees_conversion(lat=min_max_coordinates[0,1], lon=min_max_coordinates[0,0], grid_distance_m=3000)
  lon_range = np.arange(min_max_coordinates[1,0],
                    min_max_coordinates[0,0],
                    lon_spacing)
  lat_range = np.arange(min_max_coordinates[1,1],
                    min_max_coordinates[0,1],
                    lat_spacing)
  points = []
  for lon in lon_range:
    for lat in lat_range:
      point = ee.Geometry.Point(lon,lat)
      points.append(point)
  grid_points = ee.FeatureCollection(points)
  return grid_points

# Create offset points of the grid points
def create_result_grid_points(min_max_coordinates):
  lat_spacing, lon_spacing = meters_to_degrees_conversion(lat=min_max_coordinates[0,1], lon=min_max_coordinates[0,0], grid_distance_m=3000)
  lon_range = np.arange(min_max_coordinates[1,0]+lon_spacing/2,
                    min_max_coordinates[0,0],
                    lon_spacing)
  lat_range = np.arange(min_max_coordinates[1,1]+lat_spacing/2,
                    min_max_coordinates[0,1],
                    lat_spacing)
  points = []
  for lon in lon_range:
    for lat in lat_range:
      point = ee.Geometry.Point(lon,lat)
      points.append(point)
  grid_points = ee.FeatureCollection(points)
  return grid_points

# Clip grid points to the polygon
def clip_grids_points(polygon, grid, result_grid):
  grid_clip = grid.filterBounds(polygon)
  result_grid_clip = result_grid.filterBounds(polygon)
  return (grid_clip, result_grid_clip)

# Conversion of ee.FeatureCollection to a ee.Geometry list
def convert_fc_to_geometry_list(fc):
  point_geometry_list = []
  for i in range(fc.size().getInfo()):
    point_geometry_list.append(ee.Geometry.Point(fc.geometry().coordinates().get(i)))

  return point_geometry_list

# Create bounding boxes around grid points
def create_box_imagery(point):
  point_geom = ee.Geometry(point)
  box = point_geom.buffer(3500).bounds()
  return box

# Create bounding boxes around grid points
def create_box_predictions(point):
  point_geom = ee.Geometry(point)
  box = point_geom.buffer(3000).bounds()
  return box

# Create bounding box around offset points (result_grid_points)
def create_box_avg_predictions(point):
  point_geom = ee.Geometry(point)
  box = point_geom.buffer(1500).bounds()
  return box

# Get and process ASTER imagery grom google earth engine
def get_eng_img_from_point(grid_point):
  bbox = create_box_imagery(grid_point)
  image_dict = preprocessing.aster_pre_processing(ee.ImageCollection("ASTER/AST_L1T_003"), bbox)
  pp_image = image_dict['imagery']
  eng_image = band_engineering.band_engineering(pp_image)
  return eng_image

# Convert ASTER imagery to numpy array
def download_geotiff_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        return response.content
    else:
        raise ValueError(f"Failed to download the file from the URL. Status code: {response.status_code}")
    
# Convert GeoTIFF to numpy
def geotiff_to_numpy(geotiff_bytes):
    with MemoryFile(geotiff_bytes) as memfile:
        with memfile.open() as dataset:
            numpy_array = dataset.read()
            return numpy_array
        
# Convert image to numpy directly from google earth engine
def gee_image_to_numpy(grid_point):
  eng_image = get_eng_img_from_point(grid_point)
  bands = eng_image.bandNames().getInfo()
  bbox = create_box_imagery(grid_point)
  download_url = eng_image.getDownloadURL({
    'bands': bands,
    'region': bbox,
    'scale': 30,
    'format': 'GEO_TIFF'})
  geotiff_bytes = download_geotiff_from_url(download_url)
  numpy_array = geotiff_to_numpy(geotiff_bytes)
  return numpy_array

# Make predictions using model
def model_prediction(numpy_array, band_maxima):
  pre_mod_image = ui_array_processing.preprocess_array(numpy_array, band_maxima, image_size = (224, 224))
  pre_mod_image = np.transpose(pre_mod_image, (1, 2, 0))
  input_data = np.expand_dims(pre_mod_image, axis=0)
  prediction = model.predict(input_data)
  return prediction

# Put it all together to go from polygon to predictions
def PCD_prediction(polygon):
  min_max = poly_min_max_coords(polygon)
  grid = create_input_grid_points(min_max)
  result_grid = create_result_grid_points(min_max)
  grid_clip, result_clip = clip_grids_points(polygon, grid, result_grid)
  grid_list = convert_fc_to_geometry_list(grid_clip)
  res_features_pred = []
  for point in grid_list:
    numpy_array = gee_image_to_numpy(point)
    mod_prediction = model_prediction(numpy_array, band_maxima)[0][0].astype(float)
    res_box = create_box_predictions(point)
    res_feature = ee.Feature(res_box, {'prediction': mod_prediction})
    res_features_pred.append(res_feature)

  res_point_list = convert_fc_to_geometry_list(result_clip)
  res_poly_list = []
  for point in res_point_list:
    predictions = []
    for poly in res_features_pred:
      if poly.contains(point).getInfo():
        predictions.append(poly.get('prediction').getInfo())
    avg_pred = np.array(predictions).mean()
    box = create_box_avg_predictions(point)
    res_pred_f = ee.Feature(box, {'avg_prediction': avg_pred})
    res_poly_list.append(res_pred_f)

  return ee.FeatureCollection(res_poly_list)
