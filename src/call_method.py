from methods import slide_window_patcher
from methods import grid_points_water_patcher
from methods import random_points_water_patcher
from methods import random_points_bbox_patcher

def sliding_window_patcher(**kwargs):
    print("Running sliding window patcher.")
    slide_window_patcher.generate_geotiff_patches(**kwargs)

def random_water_based_patcher(**kwargs):
    print("Running random point-based patcher for water-pixels only.")
    random_points_water_patcher.generate_random_points_water_patches(**kwargs)

def grid_water_based_patcher(**kwargs):
    print("Running regular point-based patcher for water-pixels only.")
    grid_points_water_patcher.generate_grid_points_water_patches(**kwargs)

def random_polygon_based_patcher(**kwargs):
    print("Running random point-based patcher inside polygons.")
    random_points_bbox_patcher.generate_random_points_polygons_patches(**kwargs)

def call_patcher_method(method: int, patcher_args: dict):
    """Call the patcher method based on method number."""

    patchers = {
        1: sliding_window_patcher,
        2: random_water_based_patcher,
        3: grid_water_based_patcher,
        4: random_polygon_based_patcher
    }

    if method not in patchers:
        raise ValueError(f"Unknown patcher method: {method}")

    patcher_function = patchers[method]
    patcher_function(**patcher_args)