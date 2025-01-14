from pathlib import Path
import pandas as pd
import tifffile as tiff
import numpy as np
import os
from skimage import io, img_as_uint
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian

def extract_images(rootdir, 
                   projdir, 
                   panel_filename="panel.csv"):
    """
    Extract IF images from subfolders in rootdir/projdir/raw/ in one of three formats:

    1) 'stack': exactly one TIF in the folder (any name).
    2) 'order': multiple TIF files named in a numeric/alphanumeric order
       (e.g. CH01.tif, CH02.tif...), matching the number/row order of panel.csv.
    3) 'Conjugate' or 'Target': TIFs named exactly after the corresponding column
       in panel.csv (e.g., Cy3.tif, CD3.tif, etc.).
    
    Then, for each stacked TIF in analysis/1_image_out/<sample>:
      - Create a "_full" stack in analysis/3_segmentation/3a_fullstack for all channels
        where 'Full' == 1 in panel.csv.
      - Create a "_segment" stack in analysis/3_segmentation/3b_forSeg for all channels
        where 'Segment' == 1 in panel.csv.
    
    If a subfolder doesn't match the requested `extract_by` but we detect a different
    valid format, we raise a ValueError suggesting the correct format.

    Parameters
    ----------
    rootdir : str
        The root directory of your project.
    projdir : str
        A subdirectory of rootdir for the project.
    extract_by : str
        One of 'stack', 'order', 'Conjugate', or 'Target'.
    panel_filename : str
        The filename of the CSV with columns: Conjugate, Target, Full, Segment, etc.
    """

    # --------------------------------------------------------------------------
    # 1. Setup main directories
    # --------------------------------------------------------------------------
    print("Gathering Directories...")
    project_path = Path(rootdir) / projdir
    os.chdir(project_path)

    # Raw data
    raw_dir = project_path / "raw"

    # Output directories
    acquisitions_dir = project_path / "analysis" / "1_image_out"
    segment_fold_dir = project_path / "analysis" / "3_segmentation"
    output_dir = segment_fold_dir / "3a_fullstack"
    segment_dir = segment_fold_dir / "3b_forSeg"

    acquisitions_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Load panel.csv & determine channels for "Full" and "Segment"
    # --------------------------------------------------------------------------
    panel_path = project_path / panel_filename
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    panel = pd.read_csv(panel_path)

    # --------------------------------------------------------------------------
    # 4. Iterate subfolders in raw/, validate format, create single TIF
    # --------------------------------------------------------------------------
    for sample_dir in raw_dir.iterdir():
        if not sample_dir.is_dir() or sample_dir.name.startswith("."):
            continue
        
        # Create subfolder in analysis/1_image_out
        acquisition_subdir = acquisitions_dir / sample_dir.name
        acquisition_subdir.mkdir(parents=True, exist_ok=True)

        # Final stacked TIFF path
        out_tiff_path = acquisition_subdir / f"{sample_dir.name}.tiff"
        full_tiff_path = output_dir / f"{sample_dir.name}_full.tiff"
        seg_tiff_path = segment_dir / f"{sample_dir.name}_CpSeg.tiff"

        # Perform the actual stacking/copying
        # Exactly one TIF in the folder
        tif_files = list(sample_dir.glob("*.tif*"))
        if len(tif_files) != 1:
            raise ValueError(
                f"Expected exactly 1 TIF in '{sample_dir.name}', found {len(tif_files)}."
            )
        single_tif = tif_files[0]
        image = io.imread(str(single_tif))
        if len(range(image.shape[0])) != len(panel): 
            raise ValueError(
                f"Panel length is {len(panel)} and found `{len(range(image.shape[0]))}` in '{sample_dir.name}'."
            )
        io.imsave(str(out_tiff_path), image)
        
        full_stack = []
        seg_stack = []
        
        for idx in range(len(panel)):
            if panel.loc[idx, "Segment"] == 1:
                channel = image[idx, :, :]
                seg_stack.append(img_as_uint(channel))
            if panel.loc[idx, "Full"] == 1:
                channel = image[idx, :, :]
                full_stack.append(img_as_uint(channel))
        full_stack = np.stack(full_stack)
        seg_stack = np.stack(seg_stack)
        io.imsave(str(full_tiff_path), full_stack)
        io.imsave(str(seg_tiff_path), seg_stack)
    print("Done!\n")


def extract_images2(rootdir,
                   projdir,
                   panel_filename="panel.csv",
                   hotpixel=None,
                   gauss_blur=None):
    """
    Extract IF images from subfolders in rootdir/projdir/raw/ in a 'stack' format:
    - We expect exactly one TIF in each subfolder.
    - The TIF stack must match the number of rows in panel.csv.

    For each stacked TIF in analysis/1_image_out/<sample>:
      - Create a '_full' stack in analysis/3_segmentation/3a_fullstack
        for all channels where 'Full' == 1 in panel.csv.
      - Create a '_segment' stack in analysis/3_segmentation/3b_forSeg
        for all channels where 'Segment' == 1 in panel.csv.

    Optionally, remove hotpixels (if hotpixel["threshold"] is not None) and/or
    apply Gaussian blur (if gauss_blur is not None).

    Parameters
    ----------
    rootdir : str
        The root directory of your project.
    projdir : str
        A subdirectory of rootdir for the project.
    panel_filename : str
        The filename of the CSV with columns: Conjugate, Target, Full, Segment, etc.
    hotpixel : dict or None
        Dictionary with hotpixel removal parameters.
          e.g., {"threshold": 5.0, "neighborhood": 3}
        If threshold is None (or dict is None), skip hotpixel removal.
    gauss_blur : float or None
        Sigma for Gaussian blur. If None, skip blur.
    """

    def apply_gaussian_blur(img, sigma=1.0):
        """
        Applies a Gaussian blur with a given sigma.
        Returns the blurred image (preserving the original range).
        """
        # skimage.filters.gaussian returns a float64 image in [0,1] if preserve_range=False
        # So we set preserve_range=True to keep original intensity scales.
        blurred = gaussian(img, sigma=sigma, preserve_range=True)
        return blurred.astype(img.dtype)
    
    def remove_hotpixels_threshold(img, threshold=5.0, neighborhood_size=3):
        """
        Replace 'hot' pixels that are above (threshold * local_mean) with that local mean.
        """
        # Convert to float for safety
        img_float = img.astype(float)

        # Calculate local mean via uniform filter
        local_mean = uniform_filter(img_float, size=neighborhood_size)

        # Create a mask of hot pixels
        hot_mask = img_float > (threshold * local_mean)

        # Replace hot pixels with the local mean
        cleaned_img = img_float.copy()
        cleaned_img[hot_mask] = local_mean[hot_mask]

        # Convert back to original dtype (e.g., uint16) if desired
        return cleaned_img.astype(img.dtype)
    # --------------------------------------------------------------------------
    # 1. Setup main directories
    # --------------------------------------------------------------------------
    print("Gathering Directories...")
    project_path = Path(rootdir) / projdir
    os.chdir(project_path)

    # Raw data
    raw_dir = project_path / "raw"

    # Output directories
    acquisitions_dir = project_path / "analysis" / "1_image_out"
    segment_fold_dir = project_path / "analysis" / "3_segmentation"
    output_dir = segment_fold_dir / "3a_fullstack"
    segment_dir = segment_fold_dir / "3b_forSeg"

    acquisitions_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Load panel.csv & determine channels for "Full" and "Segment"
    # --------------------------------------------------------------------------
    panel_path = project_path / panel_filename
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    panel = pd.read_csv(panel_path)

    # If hotpixel is None or missing, define a default structure
    # e.g. hotpixel={"threshold": None, "neighborhood": 3}
    if hotpixel is None:
        hotpixel = {"threshold": None, "neighborhood": 3}

    # Extract the threshold and neighborhood
    hp_threshold = hotpixel.get("threshold", None)
    hp_neighborhood = hotpixel.get("neighborhood", 3)

    # --------------------------------------------------------------------------
    # 3. Iterate subfolders in raw/, validate format, create single TIF
    # --------------------------------------------------------------------------
    for sample_dir in raw_dir.iterdir():
        if not sample_dir.is_dir() or sample_dir.name.startswith("."):
            continue

        # Create subfolder in analysis/1_image_out
        acquisition_subdir = acquisitions_dir / sample_dir.name
        acquisition_subdir.mkdir(parents=True, exist_ok=True)

        # Final stacked TIFF path
        out_tiff_path = acquisition_subdir / f"{sample_dir.name}.tiff"
        full_tiff_path = output_dir / f"{sample_dir.name}_full.tiff"
        seg_tiff_path = segment_dir / f"{sample_dir.name}_CpSeg.tiff"

        # We expect exactly one TIF in the folder
        tif_files = list(sample_dir.glob("*.tif*"))
        if len(tif_files) != 1:
            raise ValueError(
                f"Expected exactly 1 TIF in '{sample_dir.name}', found {len(tif_files)}."
            )
        single_tif = tif_files[0]

        # Read the stack
        image = io.imread(str(single_tif))

        # Validate that the stack depth == number of rows in panel.csv
        if image.shape[0] != len(panel):
            raise ValueError(
                f"Panel length is {len(panel)} but found `{image.shape[0]}` channels"
                f" in '{sample_dir.name}'."
            )

        # Save the original stack (unprocessed or raw)
        io.imsave(str(out_tiff_path), image)

        # We'll build two lists of processed channels
        full_stack = []
        seg_stack = []

        # Process each channel in the stack
        for idx in range(len(panel)):
            channel = image[idx, :, :]

            # 1) Hotpixel removal if threshold is not None
            if hp_threshold is not None:
                channel = remove_hotpixels_threshold(
                    channel,
                    threshold=hp_threshold,
                    neighborhood_size=hp_neighborhood
                )

            # 2) Gaussian blur if gauss_blur is not None
            if gauss_blur is not None:
                channel = apply_gaussian_blur(channel, sigma=gauss_blur)

            # 3) Assign channels to correct stacks
            if panel.loc[idx, "Segment"] == 1:
                seg_stack.append(img_as_uint(channel))
            if panel.loc[idx, "Full"] == 1:
                full_stack.append(img_as_uint(channel))

        # Convert channel lists to stacks
        if len(full_stack) > 0:
            full_stack = np.stack(full_stack)
            io.imsave(str(full_tiff_path), full_stack)
        else:
            print(f"Warning: No 'Full' channels found for sample '{sample_dir.name}'.")

        if len(seg_stack) > 0:
            seg_stack = np.stack(seg_stack)
            io.imsave(str(seg_tiff_path), seg_stack)
        else:
            print(f"Warning: No 'Segment' channels found for sample '{sample_dir.name}'.")

    print("Done!\n")
