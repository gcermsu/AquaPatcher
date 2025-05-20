import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.coords import BoundingBox

def window_to_bbox(src, window):
    """Convert a raster window to georeferenced bounding box."""
    transform = src.window_transform(window)
    min_x, min_y = transform * (0, window.height)  # Bottom-left corner
    max_x, max_y = transform * (window.width, 0)  # Top-right corner
    return BoundingBox(min_x, min_y, max_x, max_y)

def bbox_to_window(src, bbox):
    """Convert a georeferenced bounding box (tuple) to raster window."""
    transform = src.transform
    min_x, min_y, max_x, max_y = bbox  # Unpack bbox tuple

    # Convert geographic coordinates to pixel indices
    col_min, row_max = ~transform * (min_x, max_y)  # Top-left
    col_max, row_min = ~transform * (max_x, min_y)  # Bottom-right

    # Ensure rows and columns are positive
    col_min, col_max = sorted([int(col_min), int(col_max)])
    row_min, row_max = sorted([int(row_min), int(row_max)])

    # Compute width and height
    width = col_max - col_min
    height = row_max - row_min

    return Window(col_min, row_min, width, height)

def generate_geotiff_patches(
        input_path_mask: str,
        input_path_img: str,
        output_dir: str,
        patch_size: int = 256,
        stride: int = 128,
        min_valid_pixels: int = 3,
        set_nan: bool = True
):
    """Generate 256x256 patches from a GeoTIFF mask and save valid ones as GeoTIFFs"""

    out_img = os.path.join(output_dir, "imgs")
    out_msk = os.path.join(output_dir, "masks")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    with rasterio.open(input_path_mask) as src_msk, rasterio.open(input_path_img) as src_img:
        width = src_msk.width
        height = src_msk.height
        profile_msk = src_msk.profile
        profile_img = src_img.profile

        patch_id = 0
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                patch_mask = src_msk.read(1, window=window)

                if np.count_nonzero(patch_mask == 1) >= min_valid_pixels:
                    patch_img = src_img.read(window=window)
                    transform = src_msk.window_transform(window)

                    if set_nan:
                        patch_img = np.where((patch_img == -9999) | np.isnan(patch_img), 0, patch_img)

                    # Prepare output profile
                    out_profile_maks = profile_msk.copy()
                    out_profile_maks.update({
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

                    with rasterio.open(patch_msk_out, 'w', **out_profile_maks) as dst:
                        dst.write(patch_mask, 1)

                    patch_id += 1

    print(f"Saved {patch_id} valid patches to: {output_dir}")