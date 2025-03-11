import os
import warnings
import numpy as np
import pandas as pd
import tifffile
import pyvips


# Convert Images is used to convert types and format of images
#  type can be 8bit, 16bit, 32bit, pyramidal written as ["from", "to"]
#  format can be stack or singles written as ["from", "to"]
#  images will be in im_dir with a suffix (empty if no suffix) - if stack, they'll just be a list of images, if singles, they'll be folders of the image names and individual images for panel.csv[Target] when Full==1
#  panel will have a Target and Full column. Full needs to have a 1 to match the stack or single images.
#  image.csv will need to have the image names without the suffix or ext and a column name to specify. 

def ConvertImages(
    rootdir,
    projdir, 
    image_csv_dir="image.csv",
    image_col="Image",
    panel_csv_dir="panel.csv", 
    in_dir="analysis/2_cleaned", 
    in_suffix="_cleaned",
    out_dir="analysis/2_cleaned_out", 
    out_suffix="_converted",
    type_change=["8bit", "16bit"], 
    format_change=["stack", "singles"]
):
    # Set up directory paths
    im_dir = os.path.join(rootdir, projdir, in_dir)
    im_csv_path = os.path.join(rootdir, projdir, image_csv_dir)
    panel_csv_path = os.path.join(rootdir, projdir, panel_csv_dir)
    imout_dir = os.path.join(rootdir, projdir, out_dir)
    
    # Check if CSV files exist
    if not os.path.exists(im_csv_path):
        raise FileNotFoundError(f"Image CSV not found at {im_csv_path}")
    if not os.path.exists(panel_csv_path):
        raise FileNotFoundError(f"Panel CSV not found at {panel_csv_path}")
    
    # Load CSVs
    panel = pd.read_csv(panel_csv_path)
    # Select only the channels marked as Full == 1 and reindex
    selected_markers = panel[panel['Full'] == 1].reset_index(drop=True)
    image_csv = pd.read_csv(im_csv_path)
    image_names = image_csv[image_col].tolist()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(imout_dir):
        os.makedirs(imout_dir)
    
    # Loop through each image name
    for name in image_names:
        print(f"Processing image: {name}...")
        
        # Determine input format from format_change[0]
        input_format = format_change[0].lower()
        if input_format == "stack":
            # Expect a multi‐channel file: ImageName_in_suffix.tif/.tiff
            input_file = None
            for ext in [".tif", ".tiff"]:
                candidate = os.path.join(im_dir, f"{name}{in_suffix}{ext}")
                if os.path.exists(candidate):
                    input_file = candidate
                    break
            if input_file is None:
                warnings.warn(f"Stack image for {name} not found in {im_dir}.")
                continue
            try:
                with tifffile.TiffFile(input_file) as tif:
                    image_stack = tif.asarray()
                # Assume each page is a channel – warn if the count doesn’t match
                n_channels = image_stack.shape[0] if image_stack.ndim >= 3 else 1
                if n_channels != len(selected_markers):
                    warnings.warn(f"Stack image {input_file} has {n_channels} channels; expected {len(selected_markers)}.")
            except Exception as e:
                warnings.warn(f"Error reading {input_file}: {e}")
                continue
        
        elif input_format == "singles":
            # Expect a folder: im_dir/imagename containing files named as Target_in_suffix.tif/.tiff
            folder_path = os.path.join(im_dir, name)
            if not os.path.isdir(folder_path):
                warnings.warn(f"Folder for image {name} not found in {im_dir}.")
                continue
            image_channels = []
            missing_channel = False
            for target in selected_markers["Target"]:
                channel_file = None
                for ext in [".tif", ".tiff"]:
                    candidate = os.path.join(folder_path, f"{target}{in_suffix}{ext}")
                    if os.path.exists(candidate):
                        channel_file = candidate
                        break
                if channel_file is None:
                    warnings.warn(f"Channel file for {target} not found in folder {folder_path}.")
                    missing_channel = True
                    break
                try:
                    channel_img = tifffile.imread(channel_file)
                    image_channels.append(channel_img)
                except Exception as e:
                    warnings.warn(f"Error reading {channel_file}: {e}")
                    missing_channel = True
                    break
            if missing_channel:
                continue
            # Stack individual channels into a multi‐channel array (channels first)
            image_stack = np.stack(image_channels, axis=0)
        else:
            warnings.warn("Unknown input format specified in format_change[0].")
            continue
        
        # At this stage, image_stack should be a NumPy array with shape (channels, height, width)
        # Determine current image type
        current_type = None
        if image_stack.dtype == np.uint8:
            current_type = "8bit"
        elif image_stack.dtype == np.uint16:
            current_type = "16bit"
        elif image_stack.dtype in [np.float32, np.float64]:
            current_type = "32bit"
        else:
            current_type = str(image_stack.dtype)
        
        # Warn if the actual type does not match the expected source type in type_change[0]
        src_type = type_change[0].lower()
        tgt_type = type_change[1].lower()
        if current_type != src_type:
            warnings.warn(f"Image {name} is of type {current_type} but expected {src_type}. Attempting conversion anyway.")
        
        # Perform type conversion if necessary
        if src_type == tgt_type:
            converted_stack = image_stack
        else:
            if src_type == "8bit" and tgt_type == "16bit":
                # Scale 8-bit (0-255) to 16-bit (0-65535) by multiplying with 257
                converted_stack = (image_stack.astype(np.uint16)) * 257
            elif src_type == "16bit" and tgt_type == "8bit":
                # Scale down 16-bit to 8-bit by dividing with 257
                converted_stack = (image_stack.astype(np.float32) / 257).astype(np.uint8)
            else:
                warnings.warn(f"Unsupported type conversion from {src_type} to {tgt_type}. No conversion applied.")
                converted_stack = image_stack
        
        # Output format conversion: determine target format from format_change[1]
        out_format = format_change[1].lower()
        if out_format == "stack":
            # Write a multi‐channel TIFF
            out_file = os.path.join(imout_dir, f"{name}{out_suffix}.tiff")
            try:
                tifffile.imwrite(out_file, converted_stack)
            except Exception as e:
                warnings.warn(f"Error writing stack image to {out_file}: {e}")
        
        elif out_format == "singles":
            # Create a folder for the image and save each channel separately
            image_out_folder = os.path.join(imout_dir, name)
            if not os.path.exists(image_out_folder):
                os.makedirs(image_out_folder)
            for idx, target in enumerate(selected_markers["Target"]):
                channel_img = converted_stack[idx]
                out_file = os.path.join(image_out_folder, f"{target}{out_suffix}.tiff")
                try:
                    tifffile.imwrite(out_file, channel_img)
                except Exception as e:
                    warnings.warn(f"Error writing channel {target} for image {name} to {out_file}: {e}")
        
        elif out_format == "pyramidal":
            # Export a pyramidal OME-TIFF using pyvips (if available)
            if not pyvips_available:
                warnings.warn("pyvips is not installed. Cannot export pyramidal OME-TIFF.")
            else:
                # Convert image_stack (shape: channels, height, width) to an interleaved image (width, height, bands)
                interleaved = np.transpose(converted_stack, (2, 1, 0))
                # Determine the format string for pyvips
                if tgt_type == "8bit":
                    vips_format = 'uchar'
                elif tgt_type == "16bit":
                    vips_format = 'ushort'
                else:
                    vips_format = 'float'
                width, height, bands = interleaved.shape
                try:
                    # Create a pyvips image from the raw memory
                    vips_img = pyvips.Image.new_from_memory(interleaved.tobytes(), width, height, bands, vips_format)
                    out_file = os.path.join(imout_dir, f"{name}{out_suffix}.ome.tiff")
                    # Save with tiling and pyramid options; compression is set to JPEG for demonstration.
                    vips_img.tiffsave(out_file, tile=True, pyramid=True, bigtiff=True, compression="jpeg")
                except Exception as e:
                    warnings.warn(f"Error writing pyramidal OME-TIFF for {name} to {out_file}: {e}")
        else:
            warnings.warn("Unknown output format specified in format_change[1].")
        
      
