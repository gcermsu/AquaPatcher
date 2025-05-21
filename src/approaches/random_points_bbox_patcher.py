import os
import random
import rasterio
import numpy as np
import geopandas as gpd
import rasterio.features
from sklearn.neighbors import BallTree
from shapely.geometry import Point
from rasterio.windows import Window
from shapely.geometry import Polygon
from geopy.distance import geodesic

def generate_random_points_in_polygon(polygon, n_points) -> list:
    '''Generates random points within a polygon.'''
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0

    while len(points) < n_points and attempts < n_points * 100:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
        attempts += 1

    return points

def generate_geotiff_patches_from_points(
        points_gdf: gpd.GeoDataFrame,
        input_path_mask: str,
        input_path_image: str,
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

    with rasterio.open(input_path_mask) as src_msk, rasterio.open(input_path_image) as src_img:
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

def generate_points(input_path_polygon: str, min_dist_m: float) -> gpd.GeoDataFrame:
    '''Return table of points inside the polygons'''
    gdf_polygons = gpd.read_file(input_path_polygon)
    target_crs = "EPSG:3857"

    all_points = []

    for idx, row in gdf_polygons.iterrows():
        geom: Polygon = row.geometry

        gdf_row = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf_polygons.crs)
        gdf_proj = gdf_row.to_crs(target_crs)
        area_km2 = gdf_proj.area / 1e6  # Convert m² to km²
        n_points = int((1.2 * area_km2) / 5.9)

        points = generate_random_points_in_polygon(geom, n_points)
        point_rows = [{"geometry": pt, "source_polygon_id": idx, "n_points": n_points} for pt in points]
        all_points.extend(point_rows)

    gdf_points = gpd.GeoDataFrame(all_points, crs=gdf_polygons.crs).to_crs(gdf_polygons.crs)

    # Drop closer points
    original_crs = gdf_points.crs
    gdf = gdf_points.to_crs(epsg=3857)

    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    tree = BallTree(coords)
    indices = tree.query_radius(coords, r=min_dist_m)

    to_drop = [i for i, neighbors in enumerate(indices) if len(neighbors) > 1]
    gdf_filtered = gdf.drop(index=to_drop).reset_index(drop=True).to_crs(original_crs)

    return gdf_filtered

def generate_random_points_polygons_patches(input_path_polygon: str, input_path_mask: str, input_path_image: str, output_dir: str, patch_size: int = 256, min_valid_pixels = 3, set_nan: bool = True, min_dist_m: float = 3000):
    '''Generate patches'''
    gdf_points = generate_points(input_path_polygon, min_dist_m)
    generate_geotiff_patches_from_points(gdf_points, input_path_mask, input_path_image, output_dir, patch_size, min_valid_pixels, set_nan)