import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

from skimage import img_as_float
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter

def check_extract(
    rootdir,
    projdir,
    crop=None
):
    def remove_hotpixels_threshold(img, threshold=5.0, neighborhood_size=3):
        """
        Replace 'hot' pixels that are above (threshold * local_mean) with that local mean.
        """
        img_float = img_as_float(img)
        local_mean = uniform_filter(img_float, size=neighborhood_size)
        hot_mask = img_float > (threshold * local_mean)
        cleaned_img = img_float.copy()
        cleaned_img[hot_mask] = local_mean[hot_mask]
        return cleaned_img

    def apply_gaussian_blur(img, sigma=1.0):
        """
        Applies a Gaussian blur with a given sigma.
        """
        # preserve_range=True ensures we keep original intensity scale
        blurred = gaussian(img, sigma=sigma, preserve_range=True)
        return blurred

    def process_channel(img, hp_threshold=None, hp_neighborhood=3, gauss_sigma=None):
        """
        Given a single 2D channel image:
        - If hp_threshold is not None, remove hotpixels.
        - If gauss_sigma is not None, apply Gaussian blur.
        Returns the processed image.
        """
        processed = img.copy()

        if hp_threshold is not None:
            processed = remove_hotpixels_threshold(
                processed,
                threshold=hp_threshold,
                neighborhood_size=hp_neighborhood
            )
        if gauss_sigma is not None:
            processed = apply_gaussian_blur(processed, sigma=gauss_sigma)

        return processed

    def random_crop_2D(img, crop_size=200):
        """
        Returns a random crop of shape (crop_size, crop_size) from a 2D image.
        """
        h, w = img.shape
        if crop_size > h or crop_size > w:
            raise ValueError(f"Crop size {crop_size}×{crop_size} is larger than image {h}×{w}.")
        # Random top-left corner
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        return img[y:y+crop_size, x:x+crop_size]


    """
    A UI to explore raw TIF stacks (one TIF per subfolder in raw/),
    with optional hotpixel removal, Gaussian blur, random cropping,
    and adjustable contrast.

    Parameters
    ----------
    rootdir : str
        The root directory of your project.
    projdir : str
        A subdirectory of rootdir for the project.
    crop : int or None
        If not None, will perform a random crop of size crop×crop on the selected channel.
    """

    # --------------------------------------------------------------------------
    # 1. Locate directories and panel
    # --------------------------------------------------------------------------
    project_path = os.path.join(rootdir, projdir)
    raw_dir = os.path.join(project_path, "raw")
    
    panel_path = os.path.join(project_path, "panel.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    if "Target" not in panel.columns:
        raise ValueError("panel.csv must contain a 'Target' column for channel names.")

    # We'll use the "Target" column as channel names
    channel_names = panel["Target"].tolist()
    num_channels = len(channel_names)

    # Gather subfolders in raw_dir (these are our 'images')
    subfolders = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')
    ]
    if not subfolders:
        raise ValueError(f"No subfolders found in {raw_dir}.")

    # Build the  widget
    ### Dropdown to pick which image
    image_dropdown = widgets.Dropdown(
        options=subfolders,
        description='Image:',
        layout=widgets.Layout(width='200px')
    )

    # Dropdown to pick which channel (by name in panel)
    channel_dropdown = widgets.Dropdown(
        options=channel_names,
        description='Channel:',
        layout=widgets.Layout(width='200px')
    )

    hp_threshold_text = widgets.FloatText(
        value=5.0,
        description='HP Thresh:',
        disabled=False,
        layout=widgets.Layout(width='160px')
    )

    hp_neighborhood_text = widgets.IntText(
        value=3,
        description='HP Neigh:',
        disabled=False,
        layout=widgets.Layout(width='160px')
    )
    
    gauss_sigma_text = widgets.FloatText(
        value=1.0,
        description='Gauss σ:',
        disabled=False,
        layout=widgets.Layout(width='160px')
    )

    # Contrast slider: we pick a default range [0, 1]
    # The user can drag the handles to set vmin/vmax for display.
    contrast_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=0.01,
        description='Contrast',
        layout=widgets.Layout(width='300px')
    )

    update_button = widgets.Button(
        description="Update",
        button_style='success'
    )

    output_display = widgets.Output()

    # --------------------------------------------------------------------------
    # 3. Define the callback for "Update" button
    # --------------------------------------------------------------------------
    def on_update_clicked(b):
        with output_display:
            clear_output(wait=True)
            
            # Grab user inputs
            # 1) Which image (subfolder)
            selected_image_name = image_dropdown.value
            # 2) Which channel name
            selected_channel_name = channel_dropdown.value
            # Convert that to a channel index
            try:
                channel_idx = channel_names.index(selected_channel_name)
            except ValueError:
                print(f"Channel '{selected_channel_name}' not found in panel.")
                return

            # 3) Hotpixel threshold
            hp_thresh_val = hp_threshold_text.value
            # 4) Hotpixel neighborhood
            hp_neigh_val = hp_neighborhood_text.value
            # 5) Gauss sigma
            gauss_val = gauss_sigma_text.value
            # 6) Contrast range
            vmin, vmax = contrast_slider.value

            # Build the path to the TIF file (expected one TIF in the subfolder)
            subfolder_path = os.path.join(raw_dir, selected_image_name)
            tif_files = [
                f for f in os.listdir(subfolder_path)
                if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
            ]
            if len(tif_files) != 1:
                print(f"Warning: expected 1 TIF in {subfolder_path}, found {len(tif_files)}.")
                return

            tif_path = os.path.join(subfolder_path, tif_files[0])

            # Read the stack
            stack = tiff.imread(tif_path)
            
            # Check shape
            if stack.shape[0] != num_channels:
                print(
                    f"Warning: panel has {num_channels} channels, "
                    f"but TIF has {stack.shape[0]} channels."
                )
            
            # Extract the channel of interest
            raw_channel = stack[channel_idx, :, :]

            # Optionally crop
            channel_for_processing = raw_channel
            if crop is not None:
                try:
                    channel_for_processing = random_crop_2D(raw_channel, crop_size=crop)
                except ValueError as e:
                    print(str(e))
                    return
            
            # Process the channel
            processed_channel = process_channel(
                channel_for_processing,
                hp_threshold=hp_thresh_val,
                hp_neighborhood=hp_neigh_val,
                gauss_sigma=gauss_val
            )

            # ------------------------------------------------------------------
            # Normalize each image to [0,1] for display
            # This lets vmin/vmax in [0,1] do what we expect on the slider
            # ------------------------------------------------------------------
            def safe_normalize(img):
                m = img.max()
                if m < 1e-12:
                    # Avoid divide-by-zero
                    return np.zeros_like(img, dtype=np.float32)
                return img.astype(np.float32) / m

            raw_norm = safe_normalize(channel_for_processing)
            proc_norm = safe_normalize(processed_channel)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Raw
            im0 = axes[0].imshow(raw_norm, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Raw [{selected_image_name}] - {selected_channel_name}')
            axes[0].axis('off')

            # Processed
            im1 = axes[1].imshow(proc_norm, cmap='gray', vmin=vmin, vmax=vmax)
            axes[1].set_title('Processed')
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

    update_button.on_click(on_update_clicked)

    # --------------------------------------------------------------------------
    # 4. Layout the UI
    # --------------------------------------------------------------------------
    ui = widgets.VBox([
        widgets.HBox([image_dropdown, channel_dropdown]),
        widgets.HBox([hp_threshold_text, hp_neighborhood_text, gauss_sigma_text]),
        contrast_slider,
        update_button,
        output_display
    ])
    
    display(ui)

    # Trigger an initial update to see something from the start
    on_update_clicked(None)