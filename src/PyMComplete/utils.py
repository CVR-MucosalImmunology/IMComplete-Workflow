import os
import warnings
import numpy as np
import pandas as pd
import tifffile


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
    
    if not os.path.exists(im_csv_path):
        raise FileNotFoundError(f"Image CSV not found at {im_csv_path}")
    if not os.path.exists(panel_csv_path):
        raise FileNotFoundError(f"Panel CSV not found at {panel_csv_path}")
    
    panel = pd.read_csv(panel_csv_path)
    selected_markers = panel[panel['Full'] == 1].reset_index(drop=True)
    image_csv = pd.read_csv(im_csv_path)
    image_names = image_csv[image_col].tolist()
    
    if not os.path.exists(imout_dir):
        os.makedirs(imout_dir)
    
    for name in image_names:
        print(f"Processing image: {name}...")
        
        input_format = format_change[0].lower()
        if input_format == "stack":
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
                if image_stack.ndim < 3:
                    warnings.warn(f"Stack image {input_file} has unexpected dimensions: {image_stack.shape}.")
            except Exception as e:
                warnings.warn(f"Error reading {input_file}: {e}")
                continue
        
        elif input_format == "singles":
            folder_path = os.path.join(im_dir, name)
            if not os.path.isdir(folder_path):
                warnings.warn(f"Folder for image {name} not found in {im_dir}.")
                continue
            image_channels = []
            for target in selected_markers["Target"]:
                channel_file = None
                for ext in [".tif", ".tiff"]:
                    candidate = os.path.join(folder_path, f"{target}{in_suffix}{ext}")
                    if os.path.exists(candidate):
                        channel_file = candidate
                        break
                if channel_file is None:
                    warnings.warn(f"Channel file for {target} not found in {folder_path}.")
                    break
                try:
                    channel_img = tifffile.imread(channel_file)
                    image_channels.append(channel_img)
                except Exception as e:
                    warnings.warn(f"Error reading {channel_file}: {e}")
                    break
            image_stack = np.stack(image_channels, axis=0) if image_channels else None
        else:
            warnings.warn("Unknown input format specified in format_change[0].")
            continue

        if image_stack is None:
            continue
        
        current_type = "8bit" if image_stack.dtype == np.uint8 else "16bit" if image_stack.dtype == np.uint16 else "32bit"
        src_type, tgt_type = type_change[0].lower(), type_change[1].lower()
        if current_type != src_type:
            warnings.warn(f"Image {name} is of type {current_type} but expected {src_type}. Converting anyway.")
        
        if src_type == "8bit" and tgt_type == "16bit":
            converted_stack = (image_stack.astype(np.uint16)) * 257
        elif src_type == "16bit" and tgt_type == "8bit":
            converted_stack = (image_stack.astype(np.float32) / 257).astype(np.uint8)
        else:
            converted_stack = image_stack
        
        out_format = format_change[1].lower()
        if out_format == "stack":
            out_file = os.path.join(imout_dir, f"{name}{out_suffix}.tiff")
            try:
                tifffile.imwrite(out_file, converted_stack, bigtiff=True, compression='zlib')
            except Exception as e:
                warnings.warn(f"Error writing stack image to {out_file}: {e}")
        
        elif out_format == "singles":
            image_out_folder = os.path.join(imout_dir, name)
            if not os.path.exists(image_out_folder):
                os.makedirs(image_out_folder)
            for idx, target in enumerate(selected_markers["Target"]):
                channel_img = converted_stack[idx]
                out_file = os.path.join(image_out_folder, f"{target}{out_suffix}.tiff")
                try:
                    tifffile.imwrite(out_file, channel_img, compression='zlib')
                except Exception as e:
                    warnings.warn(f"Error writing channel {target} for image {name} to {out_file}: {e}")
        
        elif out_format == "pyramidal":
            out_file = os.path.join(imout_dir, f"{name}{out_suffix}.ome.tiff")
            try:
                tifffile.imwrite(
                    out_file,
                    converted_stack,
                    bigtiff=True,
                    compression='zlib',
                    metadata={'axes': 'CYX'},
                )
            except Exception as e:
                warnings.warn(f"Error writing pyramidal OME-TIFF for {name} to {out_file}: {e}")
        else:
            warnings.warn("Unknown output format specified in format_change[1].")
