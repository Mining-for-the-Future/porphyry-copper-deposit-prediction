import utm
import ee
ee.Initialize()

def get_utm_zone_epsg_code(coordinates):
    # Check if the input is in the format [Longitude, Latitude]
    if len(coordinates) != 2:
        raise ValueError("Input coordinates should be in the format [Longitude, Latitude].")

    # Extract longitude and latitude from the input
    longitude, latitude = coordinates

    # Calculate UTM zone and hemisphere
    utm_zone = utm.latlon_to_zone_number(latitude, longitude)
    # utm_zone, utm_letter = utm.from_latlon(latitude, longitude)[2:]

    # Determine the EPSG code based on the hemisphere
    if latitude >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone

    return 'EPSG:'+str(epsg_code)

def create_box(coords, distance = 3000):
    point = ee.Geometry.Point(coords)
    proj = ee.Projection(get_utm_zone_epsg_code(coords))
    proj = proj.atScale(1) # This is necessary because the units of the projections are not necessarily a single meter. This scales all projections so that the unit is 1 m.
    buffer = point.buffer(distance, proj = proj)
    box = buffer.bounds(proj = proj)
    return box