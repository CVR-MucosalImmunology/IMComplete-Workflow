from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import tifffile as tiff
import numpy as np
import imcsegpipe
from imcsegpipe.utils import sort_channels_by_mass
import os
import csv

def bodenmiller_mcd_extract(rootdir, projdir, panel="panel.csv"):

    print("Gathering Directories...")
    os.chdir(os.path.join(rootdir, projdir))

    images_dir = Path(os.path.join(rootdir, projdir, "analysis/1_image_out")) 
    segment_fold_dir = Path(os.path.join(rootdir, projdir, "analysis/3_segmentation"))
    segment_dir = Path(os.path.join(segment_fold_dir, "3b_forSeg"))
    output_dir = Path(os.path.join(segment_fold_dir, "3a_fullstack"))

    raw = Path(os.path.join(rootdir, projdir ,"raw"))

    print("Extracting MCD...")
    # Step 1: Extract .mcd files
    temp_dirs = []
    try:
        for raw_dir in [raw]:
            zip_files = list(raw_dir.rglob("**/*.zip"))
            if len(zip_files) > 0:
                temp_dir = TemporaryDirectory()
                temp_dirs.append(temp_dir)
                for zip_file in sorted(zip_files):
                    imcsegpipe.extract_zip_file(zip_file, temp_dir.name)
        for raw_dir in [raw] + [Path(temp_dir.name) for temp_dir in temp_dirs]:
            mcd_files = list(raw_dir.rglob("*.mcd"))
            mcd_files = [i for i in mcd_files if not i.stem.startswith('.')]
            if len(mcd_files) > 0:
                txt_files = list(raw_dir.rglob("*.txt"))
                txt_files = [i for i in txt_files if not i.stem.startswith('.')]
                matched_txt_files = imcsegpipe.match_txt_files(mcd_files, txt_files)
                for mcd_file in mcd_files:
                    imcsegpipe.extract_mcd_file(
                        mcd_file,
                        images_dir / mcd_file.stem,
                        txt_files=matched_txt_files[mcd_file]
                    )
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()
        del temp_dirs
    
    samples = [
        ['Image','ImShort','ROI','ImageID','DonorID','Condition', "Crop"]
    ]

    # Read the panel.csv
    panel = pd.read_csv("panel.csv")
    print("Generating Fullstacks...")

    # Step 2: Generate image stacks (_full and _segment)
    for image_dir in images_dir.glob("[!.]*"):
        if image_dir.is_dir():
            samples.append([image_dir.name,'','','','','',''])
            imcsegpipe.create_analysis_stacks(
                acquisition_dir=image_dir,
                analysis_dir=output_dir,
                analysis_channels=sort_channels_by_mass(
                    panel.loc[panel["Full"] == 1, "Conjugate"].tolist()
                ),
                suffix="_full",
                hpf=50.0
            )
            imcsegpipe.create_analysis_stacks(
                acquisition_dir=image_dir,
                analysis_dir=segment_dir,
                analysis_channels=sort_channels_by_mass(
                    panel.loc[panel["Segment"] == 1, "Conjugate"].tolist()
                ),
                suffix="_segment",
                hpf=50.0
            )
    # Specify the file name
    filename = os.path.join(rootdir, projdir, "image.csv")

    # Writing to the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(samples)

#    # Step 3: Process TIFFs for denoising
#    if denoise:
#        print("Generating Denoise output...")
#
#        for sample_dir in images_dir.glob("[!.]*"):
#            if sample_dir.is_dir():
#                for roi_tiff_path in sample_dir.glob("*.tiff"):
#                    roi_name = roi_tiff_path.stem
#                    roi_subdir = denoise_dir / roi_name
#                    roi_subdir.mkdir(parents=True, exist_ok=True)

#                    # Load the stack using tifffile
#                    with tiff.TiffFile(roi_tiff_path) as tif:
#                        stack = tif.asarray()  # Load the entire TIFF stack as a NumPy array

#                    # Filter and unstack based on panel.csv
#                    for idx, row in panel[panel["Full"] == 1].iterrows():
#                        metal_tag = row["Conjugate"]
#                        target = row["Target"]
#                        output_name = f"{metal_tag}-{target}_{metal_tag}.tiff"
#                        output_path = roi_subdir / output_name

#                       # Extract the specific slice from the stack
#                       slice_image = stack[idx, :, :]  # Adjust indexing based on stack structure

#                        # Save the slice as a TIFF
#                        tiff.imwrite(output_path, slice_image.astype(np.uint16))  # Save as 16-bit TIFF

    print("Done!")
