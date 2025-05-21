import os
import numpy as np
import xarray as xr
import rioxarray as rxr

def _create_water_mask(swir: xr.DataArray, green: xr.DataArray) -> xr.DataArray:
    """Create mask for water pixels."""

    WATER_THRESHOLDS = {
        'swir': 0.03,
        'mndwi': 0.3
    }

    swir_mask = swir < WATER_THRESHOLDS['swir']
    mndwi = (green - swir) / (green + swir)
    mndwi_mask = mndwi > WATER_THRESHOLDS['mndwi']
    return mndwi_mask

def _find_band_file(band_paths, band_key: str):
    """Helper method to find band files based on band key."""

    BAND_MAPPING = {
        'blue': 'B02',
        'green': 'B03',
        'red': 'B04',
        'nir': 'B05',
        'swir': ['B06', 'B11'],
        'cloud': ['cloud', 'CLOUD']
    }

    target = BAND_MAPPING[band_key]
    if isinstance(target, list):
        return [band for band in band_paths if any(t in band for t in target)]
    return [band for band in band_paths if target in band]


def _load_and_preprocess_band(img_path: str, band_file) -> xr.DataArray:
    """Load and preprocess a single band."""
    if not band_file:
        return None

    band = rxr.open_rasterio(os.path.join(img_path, band_file[0])) / 10000

    if band.rio.crs is not None:
        crs_band = band.rio.crs

    return xr.where(band < 0, 0.0001, band)

def return_crs(img_path, band_file):
    band = rxr.open_rasterio(os.path.join(img_path, band_file[0])) / 10000
    return band.rio.crs

def read_images(img_path: str, bands_name):
    """Reads the images from the specified path and returns the bands."""
    band_paths = [i for i in os.listdir(img_path) if any(band in i for band in bands_name)]

    # Find band files
    green_file = _find_band_file(band_paths, 'green')
    swir_file = _find_band_file(band_paths, 'swir')
    cloud_file = _find_band_file(band_paths, 'cloud')

    crs_band = return_crs(img_path, green_file)

    # Load bands
    xda_green = _load_and_preprocess_band(img_path, green_file).rio.write_crs(crs_band)
    xda_swir = _load_and_preprocess_band(img_path, swir_file).rio.write_crs(crs_band)

    # Read cloud band
    try:
        xda_cloud = rxr.open_rasterio(img_path + '/' + cloud_file[0])
    except:
        xda_cloud = xr.DataArray(
            np.zeros(xda_green.shape),
            dims=xda_green.dims,
            coords=xda_green.coords,
            attrs=xda_green.attrs
        )
    cloud_mask = xda_cloud.rio.reproject_match(rxr.open_rasterio(os.path.join(img_path, green_file[0])))
    cloud_mask = xr.DataArray(
        np.where((cloud_mask == 2) | (cloud_mask == 3), np.nan, 0),
        coords=cloud_mask.coords,
        dims=cloud_mask.dims,
        attrs=cloud_mask.attrs
    ) # Set as nan the cloud/shadow pixels, and 0 the others.

    return xda_green,xda_swir, cloud_mask