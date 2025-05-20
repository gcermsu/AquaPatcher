# AquaPatcher

**AquaPatcher** is a Python package designed for extracting image patches from remote sensing imagery over aquatic environments. It supports multiple patch generation strategies optimized for water-based applications such as water quality monitoring and aquatic feature segmentation. The extracted patches can be used for training deep learning models, validating segmentation results, or performing statistical analysis.

You can use the unified method call_patcher_method(method, patcher_args) to invoke any of the four patching approaches. An example on how to use the package is provided in the notebook_example folder. The method is selected by an integer key (1, 2, 3, or 4), and the required arguments are passed via a dictionary.

### Available Patching Methods
- **Sliding Window Patcher (method=1)**: Uniformly extracts patches using a sliding window across the image. General purpose tiling over the full image.
Required Inputs:
  - *input_path_mask*: Path to the binary mask (single band GeoTIFF).
  - *input_path_image*: Path to the input image (multi-band GeoTIFF).
  - *output_dir*: Directory to save output patches. Creates imgs/ and masks/ subfolders.
  - *patch_size (default: 256)*: Size of the square patch in pixels.
  - *stride (default: 128)*: Step size of the sliding window.
  - *min_valid_pixels (default: 3)*: Minimum positive pixels in the patch mask.
  - *set_nan (default: True)*: Replace NaN and -9999 with 0 in the image.

- **Regular Grid Water-Based Patcher (method=2)**: Generates patches centered on a regular grid of points inside detected water bodies. Uniform sampling over water areas.
Required Inputs:
  - *input_img_files*: Dictionary of input bands to compute water mask. Requires: 'B03' for the green band, 'B06' or 'B11' for the SWIR band, and 'cloud' or 'CLOUD' for cloud mask band (All bands must be scaled by 10000 and have -9999 as no data).
  - *input_path_mask*: Path to the patch label mask (single band).
  - *input_path_image*: Path to the image to patch (multi-band).
  - *output_dir*: Directory to save patches (imgs/ and masks/).
  - *patch_size (default: 256)*: Size of the square patch in pixels.
  - *min_valid_pixels (default: 3)*: Minimum positive pixels in the patch mask.
  - *set_nan (default: True)*: Replace NaN and -9999 with 0 in the image.
  - *spacing_grid (default: 100)*: Distance between patch centers in pixels.

- **Random Water-Based Patcher (method=3)**: Randomly samples patch centers within detected water bodies. Stochastic sampling in water regions for training diversity.
Required Inputs:
    - *input_img_files*: Dictionary of input bands to compute water mask. Requires: 'B03' for the green band, 'B06' or 'B11' for the SWIR band, and 'cloud' or 'CLOUD' for cloud mask band (All bands must be scaled by 10000 and have -9999 as no data).
    - *input_path_mask*: Path to the patch label mask (single band).
    - *input_path_image*: Path to the image to patch (multi-band).
    - *output_dir*: Directory to save patches (imgs/ and masks/).
    - *patch_size (default: 256)*: Size of the square patch in pixels.
    - *min_valid_pixels (default: 3)*: Minimum positive pixels in the patch mask.
    - *set_nan (default: True)*: Replace NaN and -9999 with 0 in the image.

- **Random Polygon-Based Patcher (method=4)**: Randomly samples patch centers within user-provided polygons (e.g., lakes, river segments). Custom geographic areas of interest (AOIs).
  - *input_path_polygon*: Path to the polygon file (.gpkg) defining AOIs. Ensure the CRS of the polygon file matches the CRS of the input image.
  - *input_path_mask*: Path to the patch label mask (single band).
  - *input_path_image*: Path to the image to patch (multi-band).
  - *output_dir*: Directory to save patches (imgs/ and masks/).
  - *patch_size (default: 256)*: Size of the square patch in pixels.
  - *min_valid_pixels (default: 3)*: Minimum positive pixels in the patch mask.
  - *set_nan (default: True)*: Replace NaN and -9999 with 0 in the image.

### Dependencies management and package installation
To ensure smooth operation, the package relies on several Python libraries. To set up the environment and install the package, you can recreate the conda environment with all the necessary dependencies. This command should be run from the root of the repository:
```
conda env create -f environment.yml
```
If you prefer to use an existing conda environment, you can activate it and then install the pacereader package in development mode. This allows you to make changes to the code and test them without needing to reinstall the package. Run the following command from the root of the repository:
```
pip install -e .
```
Alternatively, you can install the package directly from GitHub using the command:
```
pip install git+https://github.com/thaimunhoz/AquaPatcher
```

### Usage example
To see examples of how to use the package, refer to the Jupyter notebook provided in the notebooks folder. The notebook contains detailed examples and usage scenarios.
