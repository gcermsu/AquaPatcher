import os
import rasterio
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio.features
from rasterio.windows import Window
from shapely.geometry import Point

from src import toolbox

def random_points_in_mask(mask_da: xr.DataArray, n_points: int, min_dist_m: float = 3000) -> gpd.GeoDataFrame:
    """Generate n random points inside a positive (1) binary raster mask."""
    mask = mask_da.values[0,:,:].astype(int)
    valid_rows, valid_cols = np.where(mask == 1)

    pixel_size = 30  # meters
    min_dist_px = min_dist_m / pixel_size

    #indices = np.random.choice(len(valid_rows), size=int(n_points), replace=False)
    #sample_rows = valid_rows[indices]
    #sample_cols = valid_cols[indices]

    valid_indices = np.arange(len(valid_rows))
    np.random.shuffle(valid_indices)

    selected_points = []
    selected_coords = []

    for idx in valid_indices:
        row = valid_rows[idx]
        col = valid_cols[idx]

        if selected_coords:
            # Check distance to previously selected points
            arr = np.array(selected_coords)
            distances = np.sqrt((arr[:, 0] - row) ** 2 + (arr[:, 1] - col) ** 2)
            if np.any(distances < min_dist_px):
                continue

        selected_coords.append((row, col))
        selected_points.append((row, col))

        if len(selected_points) == n_points:
            break

    # Convert row/col to geographic coordinates
    rows, cols = zip(*selected_points)
    transform = mask_da.rio.transform()
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    points = [Point(x, y) for x, y in zip(xs, ys)]

    # Convert row/col to x/y using the affine transform
    #transform = mask_da.rio.transform()
    #xs, ys = rasterio.transform.xy(transform, sample_rows, sample_cols)
    #points = [Point(x, y) for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(geometry=points, crs=mask_da.rio.crs)

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
        set_nan: bool = True,
        drop_edge: bool = True
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

            # Convert point (x, y) to raster indices (col, row)
            row, col = src_msk.index(point.x, point.y)

            # Calculate window upper-left corner to center on point
            half_size = patch_size // 2
            row_off = row - half_size
            col_off = col - half_size

            window = Window(col_off, row_off, patch_size, patch_size)
            patch_mask = src_msk.read(1, window=window)

            # Check valid pixels in mask
            if np.count_nonzero(patch_mask == 1) >= min_valid_pixels:
                patch_img = src_img.read(window=window)
                transform = src_msk.window_transform(window)

                # Set -9999 and nan as 0
                if set_nan:
                    patch_img = np.where((patch_img == -9999) | np.isnan(patch_img), 0, patch_img)

                # Droping small masks on the edge
                if drop_edge:
                    total_pixels = patch_img.size
                    zero_pixels = np.count_nonzero(patch_img == 0)
                    zero_ratio = zero_pixels / total_pixels
                    if zero_ratio > 0.7: continue

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

def generate_points(input_img_files: str, min_dist_m: float) -> gpd.GeoDataFrame:
    '''Return table of points inside the water mask'''
    xda_green, xda_swir, cloud_mask = toolbox.read_images(input_img_files, ['B03', 'B06', 'B11', 'cloud', 'CLOUD'])
    water_mask = toolbox._create_water_mask(xda_swir, xda_green).astype(int) # 1 - water, 0 - non water
    water_mask = water_mask + cloud_mask # Apply cloud mask to the water mask

    clean_water_mask = water_mask.fillna(0)

    pixel_count = np.sum(clean_water_mask)
    area_km2 = pixel_count * 900 / 1e6

    points_count = (1.5 * area_km2)/5.9 # 1.5 is the overlapping proportion of each patch, 5.9 is the area (in km2) of a patch size of 256x256 pixels
    gdf_points = random_points_in_mask(clean_water_mask, points_count, min_dist_m)

    return gdf_points

def generate_random_points_water_patches(input_img_files: str, input_path_mask: str, input_path_image: str, output_dir: str, patch_size: int = 256, min_valid_pixels = 3, set_nan: bool = True, min_dist_m: float = 3000, drop_edge: bool = True):
    '''Generate patches'''
    gdf_points = generate_points(input_img_files, min_dist_m)
    generate_geotiff_patches_from_points(gdf_points, input_path_mask, input_path_image, output_dir, patch_size, min_valid_pixels, set_nan, drop_edge)