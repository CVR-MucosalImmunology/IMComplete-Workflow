import os
import random
import tifffile
import numpy as np
import pandas as pd
from skimage import exposure, img_as_uint


def prep_cellpose(rootdir, projdir, nucleus="DNA", resolution=1, crop_size=200):
    # Define directories
    dir_images = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3b_forSeg")
    im_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3d_cellpose_full")
    crop_output = os.path.join(rootdir, projdir, "analysis", "3_segmentation", "3c_cellpose_crop")
    panel_file = os.path.join(rootdir, projdir, "panel.csv")

    # Load image list
    image_list = [f for f in os.listdir(dir_images) if f.endswith(('.tiff', '.tif'))]

    # Read panel.csv
    panel = pd.read_csv(panel_file)
    segmentation_targets = panel.loc[panel['Segment'] == 1, 'Target'].tolist()

    print("Segmentation Targets:", segmentation_targets)

    dna_index = [i for i, target in enumerate(segmentation_targets) if target == nucleus]
    crop_size = crop_size * resolution

    if not dna_index:
        raise ValueError("DNA channel not found in segmentation targets.")

    for image_file in image_list:
        image_path = os.path.join(dir_images, image_file)
        image = tifffile.imread(image_path)
        im_title = os.path.splitext(image_file)[0]
        # Normalize image
        normalized_stack = []
        for i in range(image.shape[0]):
            channel = image[i, :, :]
            normalized = exposure.rescale_intensity(channel, in_range='image', out_range=(0, 1))
            normalized_stack.append(img_as_uint(normalized))

        normalized_stack = np.stack(normalized_stack)  # (channels, height, width)

        # Process DNA channel
        dna_channel = normalized_stack[dna_index[0]]
        print(f"DNA channel shape: {dna_channel.shape}")  # Debug
        for idx in sorted(dna_index, reverse=True):
            normalized_stack = np.delete(normalized_stack, idx, axis=0)

        surface_mask = np.mean(normalized_stack, axis=0).astype(np.uint16)
        empty_channel = np.zeros_like(dna_channel, dtype=np.uint16)
        composite_stack = np.stack([empty_channel, surface_mask, dna_channel])

        # Save full composite image
        im_output_path = os.path.join(im_output, f"{im_title}.tiff")
        tifffile.imwrite(im_output_path, composite_stack)

        # Cropping
        height, width = composite_stack.shape[1:3]
        if width < crop_size or height < crop_size:
            crop_output_path = os.path.join(crop_output, f"{im_title}_CpCrop.tiff")
            tifffile.imwrite(crop_output_path, composite_stack)
            print(f"Image {im_title} is smaller than the cropping size. Saved without cropping.")
            continue

        workable_x = width - crop_size
        workable_y = height - crop_size
        rand_x = random.randint(0, workable_x)
        rand_y = random.randint(0, workable_y)
        cropped = composite_stack[:, rand_y:rand_y + crop_size, rand_x:rand_x + crop_size]

        crop_output_path = os.path.join(crop_output, f"{im_title}_CpCrop.tiff")
        tifffile.imwrite(crop_output_path, cropped)

    print("Done!")