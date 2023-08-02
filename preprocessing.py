# This module contains all the functions necessary to process L1T ASTER data 
# for quantitative analysis. 
# All functions are called in the function aster_pre_processing(coll,geom) which 
# takes an ee.ImageCollection object assumed to be ee.ImageCollection("ASTER/AST_L1T_003") 
# and a ee.Geometry object defining the area of interest. It returns a dictionary containing the 
# processed image as an ee.Image object. The crs and the crs transform. 


import ee
ee.Initialize()

# Filter ASTER imagery that contain all bands
def aster_bands_present_filter(collection):
    """
    Takes an image collection, assumed to be ASTER imagery.
    Returns a filtered image collection that contains only
    images with all nine VIR/SWIR bands and all 5 TIR bands.
    """
    return collection.filter(ee.Filter.And(
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B01'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B02'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B3N'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B04'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B05'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B06'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B07'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B08'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B09'),
    ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B13')
))

# Convert DN to at-sensor radiance across all bands
def aster_radiance(image):
  """
  Takes an ASTER image with pixel values in DN (as stored by Googel Earth Engine).
  Converts DN to at-sensor radiance across all bands.
  """
  coefficients = ee.ImageCollection(
        image.bandNames().map(lambda band: ee.Image(image.getNumber(ee.String('GAIN_COEFFICIENT_').cat(band))).float())
    ).toBands().rename(image.bandNames())

  radiance = image.subtract(1).multiply(coefficients)

  return image.addBands(radiance, None, True)

#Convert VIS/SWIR bands (B01 - B09) to at-sensor reflectance
def aster_reflectance(image):
  """
  Takes an ASTER image with pixel values in at-sensor radiance.
  Converts VIS/SWIR bands (B01 - B09) to at-sensor reflectance.
  """
  dayOfYear = image.date().getRelative('day', 'year')

  earthSunDistance = ee.Image().expression(
        '1 - 0.01672 * cos(0.01720209895 * (dayOfYear - 4))',
        {'dayOfYear': dayOfYear}
    )

  sunElevation = image.getNumber('SOLAR_ELEVATION')

  sunZen = ee.Image().expression(
        '(90 - sunElevation) * pi/180',
        {'sunElevation': sunElevation, 'pi': 3.14159265359}
    )

  reflectanceFactor = ee.Image().expression(
        'pi * pow(earthSunDistance, 2) / cos(sunZen)',
        {'earthSunDistance': earthSunDistance, 'sunZen': sunZen, 'pi': 3.14159265359}
    )

  irradiance = [1845.99, 1555.74, 1119.47, 231.25, 79.81, 74.99, 68.66, 59.74, 56.92]

  reflectance = image \
        .select('B01', 'B02', 'B3N', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09') \
        .multiply(reflectanceFactor) \
        .divide(irradiance)

  return image.addBands(reflectance, None, True)

# Converts TIR bands (B10 - B14) to at-satellite brightness temperature
def aster_brightness_temp(image):
  """
  Takes an ASTER image with pixel values in at-sensor radiance.
  Converts TIR bands (B10 - B14) to at-satellite brightness temperature.
  """
  K1_vals = [866.468575]
  K2_vals = [1350.069147]
  T = image.expression('K2 / (log(K1/L + 1))',
                   {'K1': K1_vals, 'K2': K2_vals, 'L': image.select('B13')}
  )

  return image.addBands(T.rename('B13'), None, True)




# Water mask
month_water = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
# I don't want to have to call this collection for every single image in the
# aster collection, so we need to call it outside of this function
def water_mask(image):
  m = image.date().get('month')
  y = image.date().get('year')
  water = month_water.filter(ee.Filter.And(
      ee.Filter.eq('month', m),
      ee.Filter.eq('year', y)
  )).mode()
  # need to add a mode() method call because the filter() method returns an image collection one image. Mode() reduces it to a single image object
  mask = water.neq(2)
  return image.updateMask(mask)

# Cloud mask
def aster_ndsi(image):
  return (image.select('B01').subtract(image.select('B04')).divide((image.select('B01').add(image.select('B04'))))).rename('ndsi')

def ac_filt1(image):
  filt1 = image.select('B02').gt(0.08)
  return image.updateMask(filt1)

def ac_filt2(image):
  filt2 = aster_ndsi(image).lt(0.7)
  return image.updateMask(filt2)

def ac_filt3(image):
  filt3 = image.select('B13').lt(300)
  return image.updateMask(filt3)

def ac_filt4(image):
  filt4 = ((image.select('B04').multiply(-1)).add(1)).multiply(image.select('B13')).lt(240)
  return image.updateMask(filt4)

def ac_filt5(image):
  filt5 = image.select('B3N').divide(image.select('B02')).lt(2)
  return image.updateMask(filt5)

def ac_filt6(image):
  filt6 = image.select('B3N').divide(image.select('B01')).lt(2.3)
  return image.updateMask(filt6)

def ac_filt7(image):
  filt7 = image.select('B3N').divide(image.select('B04')).gt(0.83)
  return image.updateMask(filt7)

def aster_cloud_mask(image):
  img = ac_filt1(image)
  img = ac_filt2(img)
  img = ac_filt3(img)
  img = ac_filt4(img)
  img = ac_filt5(img)
  img = ac_filt6(img)
  img = ac_filt7(img)
  mask = img.unmask(ee.Image.constant(-1)).eq(-1)
  return image.updateMask(mask)


# Snow mask
def aster_snow_mask(image):
  mask = aster_ndsi(image).lt(0.4)
  return image.updateMask(mask)

# Final preprocessing and masking
def aster_pre_processing(coll,geom):
  coll = coll.filterBounds(geom)
  coll = aster_bands_present_filter(coll)
  crs = coll.first().select('B01').projection().getInfo()['crs']
  transform = coll.first().select('B01').projection().getInfo()['transform']
  coll = coll.map(aster_radiance)
  coll = coll.map(aster_reflectance)
  coll = coll.map(aster_brightness_temp)
  coll = coll.map(water_mask)
  coll = coll.map(aster_cloud_mask)
  coll = coll.map(aster_snow_mask)
  coll = coll.median().clip(geom)
  return {'imagery': coll, 'crs': crs, 'transform': transform}