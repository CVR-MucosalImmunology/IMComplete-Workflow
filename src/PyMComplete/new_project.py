import os

def newProj(rootdir, project_name):
    """
    Creates a structured project folder with the given root directory and project name.

    Args:
        rootdir (str): The root directory where the project will be created.
        project_name (str): The name of the new project folder.
    """
    projdir = os.path.join(rootdir, project_name)

    # Define all required subdirectories
    acquisitions_dir = os.path.join(projdir, "analysis/1_mcd_out")
    denoise_dir = os.path.join(projdir, "analysis/2_denoise")
    segment_fold_dir = os.path.join(projdir, "analysis/3_segmentation")
    output_dir = os.path.join(segment_fold_dir, "3a_fullstack")
    segment_dir = os.path.join(segment_fold_dir, "3b_forSeg")
    crop_output = os.path.join(segment_fold_dir, "3c_cellpose_crop")
    im_output = os.path.join(segment_fold_dir, "3d_cellpose_full")
    mask_dir = os.path.join(segment_fold_dir, "3e_cellpose_mask")
    compart = os.path.join(segment_fold_dir, "3f_compartments")
    pyprof_out = os.path.join(projdir, "analysis/4_cellprofiler_output")

    # Create directories
    os.makedirs(projdir, exist_ok=True)
    os.makedirs(os.path.join(projdir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(projdir, "analysis"), exist_ok=True)
    os.makedirs(acquisitions_dir, exist_ok=True)
    os.makedirs(denoise_dir, exist_ok=True)
    os.makedirs(segment_fold_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(crop_output, exist_ok=True)
    os.makedirs(im_output, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(compart, exist_ok=True)
    os.makedirs(pyprof_out, exist_ok=True)

    print(f"Project '{project_name}' created successfully at '{projdir}'.")

