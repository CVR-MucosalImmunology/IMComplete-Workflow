import os
import random
import tifffile
import numpy as np
import pandas as pd
from skimage import exposure, img_as_uint

def prep_cellpose(rootdir, projdir, nucleus="DNA", resolution=1, crop_size=200):
    """
    Prepares images for Cellpose by:
      1) Normalizing and creating a composite stack [0=empty, 1=surface_mean, 2=DNA].
      2) Saving the full composite to 3d_cellpose_full.
      3) Cropping either by:
         - A user-specified region from image.csv (column Crop => "x_y_w_h"),
         - Or a random crop if no Crop is specified (or out of bounds).
      4) Saving the cropped version to 3c_cellpose_crop.
    """
    # Directories
    dir_images = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3b_forSeg")
    im_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3d_cellpose_full")
    crop_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3c_cellpose_crop")
    panel_file = os.path.join(rootdir, projdir, "panel.csv")
    image_csv = os.path.join(rootdir, projdir, "image.csv")  # <--- New: the CSV with Crop coords

    # Read panel.csv
    panel = pd.read_csv(panel_file)
    segmentation_targets = panel.loc[panel['Segment'] == 1, 'Target'].tolist()
    print("Segmentation Targets:", segmentation_targets)

    # Find index of the nucleus channel in the 'Segment' set
    dna_index = [i for i, target in enumerate(segmentation_targets) if target == nucleus]
    if not dna_index:
        raise ValueError("DNA channel not found in segmentation targets.")


    df_image = pd.read_csv(image_csv)
    image_list = df_image["Image"].tolist()

    # Convert the column to string so we can store or parse easily
    if "Crop" not in df_image.columns:
        raise ValueError("image.csv must contain a 'Crop' column.")

    # Convert user-specified crop_size to integer pixels with the given resolution
    crop_size_px = int(crop_size * resolution)
    df_image
    for image_file in image_list:
        image_path = os.path.join(dir_images, image_file)

        # Read the stack
        image_stack = tifffile.imread(f"{image_path}_CpSeg.tiff")
        # Normalize each channel
        normalized_stack = []
        for i in range(image_stack.shape[0]):
            channel = image_stack[i, :, :]
            normalized = exposure.rescale_intensity(channel, in_range='image', out_range=(0, 1))
            normalized_stack.append(img_as_uint(normalized))
        normalized_stack = np.stack(normalized_stack)  # shape: (C, H, W)
        
        # Identify the DNA channel
        dna_chan = normalized_stack[dna_index[0]]
        # Make the "surface_mask" = mean of all other segment channels
        # First remove the DNA channel(s) from that list:
        # e.g., if there are multiple DNA channels, remove them all
        for idx in sorted(dna_index, reverse=True):
            normalized_stack = np.delete(normalized_stack, idx, axis=0)
        # Now everything in normalized_stack is "non-DNA" segment channels
        surface_mask = np.mean(normalized_stack, axis=0).astype(np.uint16)

        # Build the composite stack [0=empty,1=surface,2=DNA]
        empty_channel = np.zeros_like(dna_chan, dtype=np.uint16)
        composite_stack = np.stack([empty_channel, surface_mask, dna_chan])

        # Save the full composite
        im_output_path = os.path.join(im_output, f"{image_file}_CpSeg.tiff")
        tifffile.imwrite(im_output_path, composite_stack)
        
        user_crop_str = df_image.loc[df_image["Image"]==image_file , "Crop"].values[0]  # e.g. "50_100_400_300" or <NA>
        _, height, width = composite_stack.shape

        if isinstance(user_crop_str, str):
            # Attempt to parse x_y_w_h
            try:
                x_str, y_str, w_str, h_str, non = user_crop_str.split("_")
                x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)

                # Check if user coords are within image bounds
                if (x + w <= width) and (y + h <= height) and (w > 0) and (h > 0):
                    cropped = composite_stack[:, y:y+h, x:x+w]
                    crop_output_path = os.path.join(crop_output, f"{image_file}_CpCrop.tiff")
                    tifffile.imwrite(crop_output_path, cropped)
                    print(f"{image_file}: used manual crop {x}_{y}_{w}_{h}")
                    continue
                else:
                    print(f"{image_file}: user crop coords out of bounds => doing random crop.")
            except Exception as e:
                print(f"{image_file}: error parsing Crop='{user_crop_str}' => doing random crop. {e}")
                
        # If we get here, no valid user crop => do random crop
        if width < crop_size_px or height < crop_size_px:
            # The image is smaller than the desired crop
            crop_output_path = os.path.join(crop_output, f"{image_file}_CpCrop.tiff")
            tifffile.imwrite(crop_output_path, composite_stack)
            print(f"Image {image_file} is smaller than {crop_size_px} px => saved without cropping.")
            continue

        workable_x = width - crop_size_px
        workable_y = height - crop_size_px
        rand_x = random.randint(0, workable_x)
        rand_y = random.randint(0, workable_y)
        cropped = composite_stack[:, rand_y:rand_y + crop_size_px, rand_x:rand_x + crop_size_px]
        
        coords_str = f"{rand_x}_{rand_y}_{crop_size_px}_{crop_size_px}_random"
        
        df_image.loc[df_image["Image"]==image_file, "Crop"] = coords_str

        crop_output_path = os.path.join(crop_output, f"{image_file}_CpCrop.tiff")
        tifffile.imwrite(crop_output_path, cropped)
        print(f"{image_file}: random-cropped at (x={rand_x}, y={rand_y}, size={crop_size_px}).")

    df_image.to_csv(os.path.join(rootdir, projdir, "image.csv"), index=False)
    print("Done!")

