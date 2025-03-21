import os
import csv

def NewProject(rootdir, projdir):
    """
    Creates a structured project folder with the given root directory and project name.

    Args:
        rootdir (str): The root directory where the project will be created.
        project_name (str): The name of the new project folder.
    """
    if os.path.isdir(os.path.join(rootdir,projdir)):
        print("Directory already exists! Check the contents and continue, or choose a new Project name.") 
        return

    # Define all required subdirectories
    acquisitions_dir = os.path.join(rootdir, projdir, "analysis/1_image_out")
    cleaned_dir = os.path.join(rootdir, projdir, "analysis/2_cleaned")
    segment_fold_dir = os.path.join(rootdir, projdir, "analysis/3_segmentation")
    crop_output = os.path.join(segment_fold_dir, "3a_cellpose_crop")
    im_output = os.path.join(segment_fold_dir, "3b_cellpose_full")
    mask_dir = os.path.join(segment_fold_dir, "3c_cellpose_mask")
    compart = os.path.join(segment_fold_dir, "3d_compartments")
    pyprof_out = os.path.join(rootdir, projdir, "analysis/4_pyprofiler_output")
    R_out = os.path.join(rootdir, projdir, "analysis/5_R_analysis")
    #meta_out = os.path.join(rootdir, projdir, ".meta")

    # Create directories
    os.makedirs(os.path.join(rootdir, projdir), exist_ok=True)
    os.makedirs(os.path.join(rootdir, projdir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(rootdir, projdir, "analysis"), exist_ok=True)
    os.makedirs(acquisitions_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(segment_fold_dir, exist_ok=True)
    os.makedirs(crop_output, exist_ok=True)
    os.makedirs(im_output, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(compart, exist_ok=True)
    os.makedirs(pyprof_out, exist_ok=True)
    os.makedirs(R_out, exist_ok=True)

    print(f"Project '{projdir}' created successfully.")

    # Data to be written to the CSV file
    panel = [
        ["Conjugate", "Target", "Full", "Segment"]
    ]

    # Specify the file name
    filename = os.path.join(rootdir, projdir, "panel.csv")

    # Writing to the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(panel)