import os
import random
import numpy as np
import pandas as pd
from skimage import io, exposure, img_as_uint

def prep_cellpose(rootdir, projdir, dna = "DNA", extra=1, square_size=200):

    # Define directories
    dir_images = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3b_forSeg")
    im_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3d_cellpose_full")
    crop_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3c_cellpose_crop")
    panel_file = os.path.join(rootdir, projdir, "panel.csv")

    # Check for extra folder condition
    if extra:
        im_extra_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3d_cellpose_full_extra")
        os.makedirs(im_extra_output, exist_ok=True)

    # Load image list
    image_list = [f for f in os.listdir(dir_images) if f.endswith(('.tiff', '.tif'))]

        # Read panel.csv
    panel = pd.read_csv(panel_file)
    segmentation_targets = panel.loc[panel['Segment'] == 1, 'Target'].tolist()

    print("Segmentation Targets:", segmentation_targets)

    dna_index = [i for i, target in enumerate(segmentation_targets) if target == dna]

    for image_file in image_list:
        image_path = os.path.join(dir_images, image_file)
        image = io.imread(image_path)
        im_title = os.path.splitext(image_file)[0]

        # Normalize image
        normalized_stack = []
        
        for i in range(image.shape[0]):
            channel = image[i, :, :]
            normalized = exposure.rescale_intensity(channel, in_range='image', out_range=(0, 1))
            normalized_stack.append(img_as_uint(normalized))
        normalized_stack = np.stack(normalized_stack)

        # Process DNA channel
        if dna_index:
            dna_channel = normalized_stack[dna_index[0]]
            for idx in sorted(dna_index, reverse=True):
                normalized_stack = np.delete(normalized_stack, idx, axis=0)
        else:
            raise ValueError("DNA channel not found in segmentation targets.")

        surface_mask = np.mean(normalized_stack, axis=0).astype(np.uint16)
        empty_channel = np.zeros_like(dna_channel, dtype=np.uint16)
        composite_stack = np.stack([empty_channel, surface_mask, dna_channel])

            # Save full composite image
        im_output_path = os.path.join(im_output, f"{im_title}_CpSeg.tiff")
        io.imsave(im_output_path, composite_stack)

            # Cropping
        height, width = composite_stack.shape[1:3]
        if width < square_size or height < square_size:
            crop_output_path = os.path.join(crop_output, f"{im_title}_CpCrop.tiff")
            io.imsave(crop_output_path, composite_stack)
            print(f"Image {im_title} is smaller than the cropping size. Saved without cropping.")
            continue

        workable_x = width - square_size
        workable_y = height - square_size
        rand_x = random.randint(0, workable_x)
        rand_y = random.randint(0, workable_y)
        cropped = composite_stack[:, rand_y:rand_y + square_size, rand_x:rand_x + square_size]
        crop_output_path = os.path.join(crop_output, f"{im_title}_CpCrop.tiff")
        io.imsave(crop_output_path, cropped)

            # Extra folder operations
        if extra:
            extra_image_output = os.path.join(im_extra_output, im_title)
            os.makedirs(extra_image_output, exist_ok=True)

                # Save each slice as individual TIFF
            for idx, target in enumerate(segmentation_targets):
                slice_path = os.path.join(extra_image_output, f"{target}.tiff")
                io.imsave(slice_path, normalized_stack2[idx])

                # Create and save stackplusone
            stackplusone = []
        
            for i in range(image.shape[0]):
                channel = image[i, :, :]
                normalized = exposure.rescale_intensity(channel, in_range='image', out_range=(0, 1))
                stackplusone.append(img_as_uint(normalized))

            stackplusone.append(img_as_uint(surface_mask)) # add the surface mask
            stackplusone = np.stack(stackplusone)
            for idx, target in enumerate(segmentation_targets):
                slice_path = os.path.join(extra_image_output, f"{target}.tiff")
                io.imsave(slice_path, stackplusone[idx])

            stackplusone_path = os.path.join(extra_image_output, "stackplusone.tiff")
            io.imsave(stackplusone_path, stackplusone)

    print("Done!")
