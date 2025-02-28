import os
import random
import tifffile
import numpy as np
import pandas as pd
from skimage import exposure, img_as_uint

def PrepCellpose(
    rootdir, 
    projdir, 
    nucleus="DNA", 
    resolution=1, 
    crop_size=200,
    panel_dir="panel.csv",
    images_dir="image.csv",
    im_from="analysis/2_cleaned", 
    suffix="_cleaned",
    fullstack=True,
    crop_to="analysis/3_segmentation/3a_cellpose_crop",
    full_to="analysis/3_segmentation/3b_cellpose_full",
    out_suffix = "_CpSeg"
): 
    # Define directories
    panel_file = os.path.join(rootdir, projdir, panel_dir)
    image_csv = os.path.join(rootdir, projdir, images_dir) 
    dir_images = os.path.join(rootdir, projdir, im_from)
    crop_output = os.path.join(rootdir, projdir, crop_to)
    im_output = os.path.join(rootdir, projdir, full_to)

    # Read panel.csv
    panel = pd.read_csv(panel_file)

    # Read image.csv and check for required column
    df_image = pd.read_csv(image_csv)
    if "Crop" not in df_image.columns:
        raise ValueError("image.csv must contain a 'Crop' column.\n")

    # Get list of images to process
    image_list = df_image["Image"].tolist()

    # Convert user-specified crop_size (in resolution units) to pixels
    crop_size_px = int(crop_size * resolution)

    # Process each image
    for image_file in image_list:
        # Build the expected filename using the suffix
        candidate1 = os.path.join(dir_images, f"{image_file}{suffix}.tiff")
        candidate2 = os.path.join(dir_images, f"{image_file}{suffix}.tif")
        if os.path.exists(candidate1):
            full_image_path = candidate1
        elif os.path.exists(candidate2):
            full_image_path = candidate2
        else:
            raise FileNotFoundError(
                f"Error: File '{image_file}{suffix}.tiff' (or .tif) not found in {dir_images}. "
                "Please check your suffix and naming.\n"
            )

        # Read the image stack from file
        image_stack = tifffile.imread(full_image_path)

        # Process based on the fullstack flag
        if fullstack:
            # Expect the stack to have only the channels where Full == 1
            expected_channels = int(panel["Full"].sum())
            if image_stack.shape[0] != expected_channels:
                raise ValueError(
                    f"Error: For fullstack==True, expected {expected_channels} channels (per panel.csv 'Full' column) "
                    f"but found {image_stack.shape[0]} in image '{image_file}{suffix}'.\n"
                )
            # Create a sub-panel for the full channels
            full_panel = panel[panel["Full"] == 1].reset_index(drop=True)
        else:
            # Expect the stack to have all channels (length == panel length)
            if image_stack.shape[0] != len(panel):
                raise ValueError(
                    f"Error: For fullstack==False, expected image stack length equal to panel length ({len(panel)}) "
                    f"but found {image_stack.shape[0]} channels in image '{image_file}{suffix}'.\n"
                )
            # Subset the stack to only the channels where Full == 1
            full_indices = panel.index[panel["Full"] == 1].tolist()
            image_stack = image_stack[full_indices, :, :]
            full_panel = panel.loc[panel["Full"] == 1].reset_index(drop=True)

        # Determine segmentation targets from the full_panel (only for channels flagged for segmentation)
        segmentation_targets = full_panel.loc[full_panel["Segment"] == 1, "Target"].tolist()
        print("Segmentation Targets for image", image_file, ":", segmentation_targets,".\n")

        # Find the index of the nucleus channel within the segmentation targets
        dna_index = [i for i, target in enumerate(segmentation_targets) if target == nucleus]
        if not dna_index:
            raise ValueError(
                f"Error: DNA channel '{nucleus}' not found in segmentation targets for image '{image_file}{suffix}'.\n"
            )
        # Normalise only the channels flagged for segmentation (Segment == 1)
        normalized_stack = []
        for i in range(image_stack.shape[0]):
            # Check if the current channel should be segmented according to panel
            if full_panel.iloc[i]["Segment"] == 1:
                channel = image_stack[i, :, :]
                normalized = exposure.rescale_intensity(channel, in_range='image', out_range=(0, 1))
                normalized_stack.append(img_as_uint(normalized))
        if normalized_stack:
            normalized_stack = np.stack(normalized_stack)  # shape: (C_segment, H, W)
        else:
            raise ValueError("No channels with Segment==1 found in panel.")
        
        # Identify the DNA channel from the normalised stack
        dna_chan = normalized_stack[dna_index[0]]
        # Remove the DNA channel(s) to compute the surface mask from the remaining channels
        for idx in sorted(dna_index, reverse=True):
            normalized_stack = np.delete(normalized_stack, idx, axis=0)
        surface_mask = np.mean(normalized_stack, axis=0).astype(np.uint16)

        # Build the composite stack with three channels: [empty, surface, DNA]
        empty_channel = np.zeros_like(dna_chan, dtype=np.uint16)
        composite_stack = np.stack([empty_channel, surface_mask, dna_chan])

        # Save the full composite image
        im_output_path = os.path.join(im_output, f"{image_file}{out_suffix}.tiff")
        tifffile.imwrite(im_output_path, composite_stack)

        # Determine cropping parameters
        user_crop_str = df_image.loc[df_image["Image"] == image_file, "Crop"].values[0]  # e.g., "50_100_400_300" or <NA>
        _, height, width = composite_stack.shape

        if isinstance(user_crop_str, str) and user_crop_str.lower() != "nan":
            try:
                parts = user_crop_str.split("_")
                if len(parts) < 4:
                    raise ValueError("Not enough crop parameters provided.\n")
                x, y, w, h = map(int, parts[:4])
                # Validate that the provided crop coordinates are within image bounds
                if (x + w <= width) and (y + h <= height) and (w > 0) and (h > 0):
                    cropped = composite_stack[:, y:y+h, x:x+w]
                    crop_output_path = os.path.join(crop_output, f"{image_file}{out_suffix}.tiff")
                    tifffile.imwrite(crop_output_path, cropped)
                    print(f"{image_file}: used manual crop {x}_{y}_{w}_{h}.\n")
                    continue
                else:
                    print(f"{image_file}: user crop coordinates out of bounds => performing random crop.\n")
            except Exception as e:
                print(f"{image_file}: error parsing Crop='{user_crop_str}' => performing random crop. {e}\n")

        # If no valid user crop is provided, perform a random crop
        if width < crop_size_px or height < crop_size_px:
            # If the image is smaller than the desired crop size, save it without cropping.
            crop_output_path = os.path.join(crop_output, f"{image_file}{out_suffix}.tiff")
            tifffile.imwrite(crop_output_path, composite_stack)
            print(f"Image {image_file} is smaller than {crop_size_px} px => saved without cropping.\n")
            continue

        workable_x = width - crop_size_px
        workable_y = height - crop_size_px
        rand_x = random.randint(0, workable_x)
        rand_y = random.randint(0, workable_y)
        cropped = composite_stack[:, rand_y:rand_y + crop_size_px, rand_x:rand_x + crop_size_px]
        
        coords_str = f"{rand_x}_{rand_y}_{crop_size_px}_{crop_size_px}_random"
        df_image.loc[df_image["Image"] == image_file, "Crop"] = coords_str

        crop_output_path = os.path.join(crop_output, f"{image_file}{out_suffix}.tiff")
        tifffile.imwrite(crop_output_path, cropped)
        print(f"{image_file}: random-cropped at (x={rand_x}, y={rand_y}, size={crop_size_px})\n.")

    # Write the updated image.csv back to disk
    df_image.to_csv(os.path.join(rootdir, projdir, images_dir), index=False)
    print("\nDone!\n")

