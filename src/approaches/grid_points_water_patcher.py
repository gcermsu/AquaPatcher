import os
import rasterio
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio.features
from shapely.geometry import Point
from rasterio.windows import Window

from src import toolbox

def regular_points_in_mask(mask_da: xr.DataArray, spacing: int) -> gpd.GeoDataFrame:
    """Generate regularly spaced points inside a binary mask (value == 1)."""
    mask = mask_da.values[0, :, :]  # (rows, cols)
    transform = mask_da.rio.transform()
    crs = mask_da.rio.crs
    height, width = mask.shape

    rows = np.arange(0, height, spacing)
    cols = np.arange(0, width, spacing)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    grid_rows = grid_rows.flatten()
    grid_cols = grid_cols.flatten()

    valid_mask = mask[grid_rows, grid_cols] == 1
    valid_rows = grid_rows[valid_mask]
    valid_cols = grid_cols[valid_mask]

    xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)
    points = [Point(x, y) for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    gdf["x"] = xs
    gdf["y"] = ys

    return gdf

def generate_geotiff_patches_from_points(
        points_gdf: gpd.GeoDataFrame,
        input_path_mask: str,
        input_path_img: str,
        output_dir: str,
        patch_size: int = 256,
        min_valid_pixels: int = 3,
        set_nan: bool = True
):
    """Generate patches centered on GeoDataFrame points and save valid ones as GeoTIFFs."""
    out_img = os.path.join(output_dir, "imgs")
    out_msk = os.path.join(output_dir, "masks")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    with rasterio.open(input_path_mask) as src_msk, rasterio.open(input_path_img) as src_img:
        profile_msk = src_msk.profile
        profile_img = src_img.profile

        patch_id = 0
        for _, row in points_gdf.iterrows():
            point = row.geometry

            row, col = src_msk.index(point.x, point.y)

            half_size = patch_size // 2
            row_off = row - half_size
            col_off = col - half_size

            window = Window(col_off, row_off, patch_size, patch_size)
            patch_mask = src_msk.read(1, window=window)

            if np.count_nonzero(patch_mask == 1) >= min_valid_pixels:
                patch_img = src_img.read(window=window)
                transform = src_msk.window_transform(window)

                if set_nan:
                    patch_img = np.where((patch_img == -9999) | np.isnan(patch_img), 0, patch_img)

                out_profile_mask = profile_msk.copy()
                out_profile_mask.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': transform
                })

                out_profile_img = profile_img.copy()
                out_profile_img.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': transform,
                    'count': patch_img.shape[0]
                })

                patch_img_out = os.path.join(out_img, f"img_{patch_id}.tif")
                patch_msk_out = os.path.join(out_msk, f"mask_{patch_id}.tif")

                with rasterio.open(patch_img_out, 'w', **out_profile_img) as dst:
                    dst.write(patch_img)

                with rasterio.open(patch_msk_out, 'w', **out_profile_mask) as dst:
                    dst.write(patch_mask, 1)

                patch_id += 1

    print(f"Saved {patch_id} valid patches to: {output_dir}")

def generate_points(input_img_files: str, spacing_grid: int) -> gpd.GeoDataFrame:
    '''Return table of points inside the water mask'''
    xda_green, xda_swir, cloud_mask = toolbox.read_images(input_img_files, ['B03', 'B06', 'B11', 'cloud', 'CLOUD'])
    water_mask = toolbox._create_water_mask(xda_swir, xda_green).astype(int) # 1 - water, 0 - non water
    water_mask = water_mask + cloud_mask # Apply cloud mask to the water mask
    clean_water_mask = water_mask.fillna(0)
    gdf_points = regular_points_in_mask(clean_water_mask, spacing_grid)
    return gdf_points

def generate_grid_points_water_patches(input_img_files: str, input_path_mask: str, input_path_image: str, output_dir: str, patch_size: int = 256, min_valid_pixels = 3, set_nan: bool = True, spacing_grid: int = 100):
    '''Generate patches'''
    gdf_points = generate_points(input_img_files, spacing_grid)
    generate_geotiff_patches_from_points(gdf_points, input_path_mask, input_path_image, output_dir, patch_size, min_valid_pixels, set_nan)