import os
import tifffile as tiff
import numpy as np


def denoise_convert(rootdir: str, projdir: str, denoiseddir: str) -> None:
    """
    Iterates through all TIFF images in the given denoised directory, applies metadata, and overwrites the image.

    Parameters:
    rootdir (str): Root directory path.
    projdir (str): Project directory path relative to root.
    denoiseddir (str): Folder containing the denoised images.
    """

    # Construct the full directory path
    directory = os.path.join(rootdir, projdir, denoiseddir)

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Process only TIFF files
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            input_file = os.path.join(directory, filename)

            try:
                # Read the multi-frame TIFF file
                with tiff.TiffFile(input_file) as tif_file:
                    data = tif_file.asarray()  # Load the image as a NumPy array

                # Set properties
                metadata = {
                    'axes': 'CYX',  # Define dimensions as Channels (C), Height (Y), Width (X)
                    'unit': 'um',   # Units are in micrometers
                    'spacing': 1.0,  # Z-spacing (voxel depth)
                    'resolution': (1.0, 1.0),  # Pixel resolution: 1 µm per pixel
                    'physical_size_x': 1.0,   # Pixel width in µm
                    'physical_size_y': 1.0    # Pixel height in µm
                }

                # Overwrite the image with updated metadata
                tiff.imwrite(
                    input_file,  # Overwrite original image
                    data,
                    metadata=metadata,
                    imagej=True,  # Ensures compatibility with ImageJ
                    resolution=(1.0, 1.0)  # DPI resolution (1 µm per pixel)
                )

                print(f"Processed and saved: {input_file}")

            except Exception as e:
                print(f"Failed to process {input_file}: {e}")


# Example usage
# denoise_convert('path/to/rootdir', 'project_dir', 'denoised_dir')
