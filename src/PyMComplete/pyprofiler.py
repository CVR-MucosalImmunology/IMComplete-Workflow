import os
import time
import numpy as np
import pandas as pd
from tifffile import imread
from scipy.ndimage import center_of_mass, find_objects
from skimage.measure import find_contours
import torch

def pyprofiler(rootdir = "", 
               projdir = "", 
               mean = 1, 
               shape = 1, 
               geometry = 1, 
               compartment = 1,
               mask_dir =  "analysis/3_segmentation/3e_cellpose_mask",
               image_dir = "analysis/3_segmentation/3a_fullstack", 
               compartment_dir =  "analysis/3_segmentation/3f_compartments", 
               out_dir = "analysis/4_pyprofiler_output/cell.csv"):

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set working directory
    #os.chdir(os.path.join(rootdir, projdir))

    # Define directories for masks, stacks, and compartments
    masks_dir = os.path.join(rootdir,projdir, mask_dir)
    stacks_dir = os.path.join(rootdir,projdir, image_dir)
    compartments_dir = os.path.join(rootdir,projdir, compartment_dir)

    # Get list of masks and stacks
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.tif', '.tiff'))]
    stack_files = [f for f in os.listdir(stacks_dir) if f.endswith(('.tif', '.tiff'))]

    # Match mask and stack files by name
    image_names = [os.path.splitext(f)[0].replace("_CpSeg_mask", "") for f in mask_files]

    # Identify compartments if applicable
    if compartment:
        compartment_folders = [f for f in os.listdir(compartments_dir) if os.path.isdir(os.path.join(compartments_dir, f))]
        if not compartment_folders:
            print("No folders found in the compartments directory. Disabling compartment processing.")
            compartment = 0
        else:
            compartment_masks = {}
            for folder in compartment_folders:
                folder_path = os.path.join(compartments_dir, folder)
                compartment_masks[folder] = {
                    name: os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(('.tif', '.tiff'))
                }

    # Process each image
    all_results = []
    start_time = time.time()

    for name in image_names:
        print(f"Processing {name}...")

        # Load mask and stack
        mask_path = os.path.join(masks_dir, f"{name}_CpSeg_mask.tif")
        stack_path = os.path.join(stacks_dir, f"{name}_full.tiff")

        cell_mask = imread(mask_path)  # Single-layer TIFF mask
        fluorescence_stack = imread(stack_path)  # Multi-layer TIFF

        # Convert to PyTorch tensors
        cell_mask_tensor = torch.tensor(cell_mask, device=device, dtype=torch.int32)
        fluorescence_stack_tensor = torch.tensor(fluorescence_stack, device=device, dtype=torch.float32)

        # Load panel.csv file
        panel_path = os.path.join(rootdir,projdir, "panel.csv")
        panel = pd.read_csv(panel_path)
        selected_markers = panel[panel['Full'] == 1].reset_index(drop=True)
        selected_indices = range(len(selected_markers))  # Indices correspond to stack order
        selected_names = selected_markers['Target'].values  # Names of relevant markers

        # Extract unique CellIDs (include CellID 0 for background)
        cell_ids = torch.unique(cell_mask_tensor).cpu().numpy()

        # Initialize results for this image
        results = []

        # Check for compartment masks
        missing_compartments = []
        compartment_data = {}
        if compartment:
            for comp_name, comp_files in compartment_masks.items():
                comp_file = comp_files.get(f"{name}_compartment.tiff")
                if not comp_file:
                    missing_compartments.append(comp_name)
                else:
                    compartment_data[comp_name] = torch.tensor(imread(comp_file), device=device, dtype=torch.float32)

            if missing_compartments:
                action = input(f"Missing compartments {missing_compartments} for {name}. Choose action: \n1) Continue with NA\n2) Stop\n3) Ignore compartment data\nEnter choice: ")
                if action == "2":
                    print("Stopping execution. Please address missing compartments.")
                    exit()
                elif action == "3":
                    compartment = 0
                    compartment_data = {}

        # Process each cell
        for cell_id in cell_ids:
            cell_region = (cell_mask_tensor == cell_id)
            cell_data = {"Image": name, "CellID": int(cell_id)}

            if shape:
                # Area
                cell_data["Area"] = int(torch.sum(cell_region).item())

                # Centroid
                if torch.any(cell_region):
                    indices = torch.nonzero(cell_region, as_tuple=True)
                    centroid_y = torch.mean(indices[0].float()).item()
                    centroid_x = torch.mean(indices[1].float()).item()
                else:
                    centroid_y, centroid_x = np.nan, np.nan
                cell_data["CentroidX"] = centroid_x
                cell_data["CentroidY"] = centroid_y

                # Bounding box and Eccentricity
                bbox_indices = torch.nonzero(cell_region)
                if bbox_indices.shape[0] > 0:
                    height = bbox_indices[:, 0].max().item() - bbox_indices[:, 0].min().item() + 1
                    width = bbox_indices[:, 1].max().item() - bbox_indices[:, 1].min().item() + 1
                    cell_data["Eccentricity"] = height / width if width != 0 else np.nan
                else:
                    cell_data["Eccentricity"] = np.nan
            if geometry:
                # Extract contours
                cell_region_cpu = cell_region.cpu().numpy()
                contours = find_contours(cell_region_cpu, level=0.5)
                geom = []
                for contour in contours:
                    geom.append(contour.tolist())
                cell_data["Geometry"] = geom

            if mean:
                for idx, marker in zip(selected_indices, selected_names):
                    fluorescence_slice = fluorescence_stack_tensor[idx]
                    mean_intensity = torch.mean(fluorescence_slice[cell_region].float()).item() / 65535  # Normalize to 16-bit range
                    cell_data[marker] = mean_intensity

            if compartment:
                for comp_name, comp_mask in compartment_data.items():
                    comp_region = comp_mask[cell_region]
                    comp_mean = torch.sum(comp_region).item() / cell_data["Area"] if cell_data["Area"] > 0 else np.nan
                    cell_data[comp_name] = comp_mean

            # Append cell data to results
            results.append(cell_data)

        # Append image results to all results
        all_results.extend(results)

    # Create a DataFrame from all results
    final_df = pd.DataFrame(all_results)

    # Save to CSV
    output_path = os.path.join(rootdir,projdir, out_dir)
    final_df.to_csv(output_path, index=False)

    print("Processing complete.")
    print("Total time taken:", time.time() - start_time)
    print(f"Results saved to {output_path}")
