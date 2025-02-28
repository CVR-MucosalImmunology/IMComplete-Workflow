import os
import time
import numpy as np
import pandas as pd

from tifffile import imread
from skimage.measure import find_contours
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial.distance import cdist

from collections import defaultdict

def PyProfiler(rootdir = "", 
               projdir = "", 
               mean = 1, 
               shape = 1, 
               geometry = 1, 
               compartment = 1,
               compartment_measure = "mean",
               panel_path = "panel.csv",
               mask_dir =  "analysis/3_segmentation/3e_cellpose_mask",
               image_dir = "analysis/3_segmentation/3a_fullstack", 
               compartment_dir =  "analysis/3_segmentation/3f_compartments", 
               out_dir = "analysis/4_pyprofiler_output/cell.csv",
               geom_out_dir = "analysis/4_pyprofiler_output/geom.csv", 
               mask_suffix = "_mask",
               image_suffix = "_full",
               comp_suffix = "_compartment",
               neighbours=0,           # Number of nearest neighbours to find
               boundary_contacts=False # If True, store boundary contact breakdown in a single column
               ):

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    masks_dir = os.path.join(rootdir, projdir, mask_dir)
    stacks_dir = os.path.join(rootdir, projdir, image_dir)
    compartments_dir = os.path.join(rootdir, projdir, compartment_dir)

    # Get list of mask files (.tif or .tiff)
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.tif', '.tiff'))]
    image_names = [os.path.splitext(f)[0].replace(mask_suffix, "") for f in mask_files]

    # Identify compartments if applicable
    if compartment:
        compartment_folders = [
            f for f in os.listdir(compartments_dir) 
            if os.path.isdir(os.path.join(compartments_dir, f))
        ]
        if not compartment_folders:
            print("No folders found in the compartments directory. Disabling compartment processing.")
            compartment = 0
        else:
            compartment_masks = {}
            for folder in compartment_folders:
                folder_path = os.path.join(compartments_dir, folder)
                compartment_masks[folder] = {
                    f: os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(('.tif', '.tiff'))
                }

    # Prepare for overall results
    all_results = []
    all_geom_results = []
    start_time = time.time()

    # ----------------------------------------------------------
    # Process each image
    # ----------------------------------------------------------
    for name in image_names:
        print(f"Processing {name}...")

        # Resolve actual mask path
        mask_path_tif = os.path.join(masks_dir, f"{name}{mask_suffix}.tif")
        mask_path_tiff = os.path.join(masks_dir, f"{name}{mask_suffix}.tiff")
        if os.path.exists(mask_path_tif):
            mask_path = mask_path_tif
        elif os.path.exists(mask_path_tiff):
            mask_path = mask_path_tiff
        else:
            print(f"No mask file found with .tif or .tiff extension for {name}")
            continue

        # Resolve actual stack path
        stack_path_tif = os.path.join(stacks_dir, f"{name}{image_suffix}.tif")
        stack_path_tiff = os.path.join(stacks_dir, f"{name}{image_suffix}.tiff")
        if os.path.exists(stack_path_tif):
            stack_path = stack_path_tif
        elif os.path.exists(stack_path_tiff):
            stack_path = stack_path_tiff
        else:
            print(f"No stack file found with .tif or .tiff extension for {name}")
            continue

        # Read images
        cell_mask = imread(mask_path)           
        fluorescence_stack = imread(stack_path) 

        # Convert to PyTorch
        cell_mask_tensor = torch.tensor(cell_mask, device=device, dtype=torch.int32)
        fluorescence_stack_tensor = torch.tensor(fluorescence_stack, device=device, dtype=torch.float32)

        # If boundary contacts are needed, keep a CPU copy
        cell_mask_cpu = cell_mask_tensor.cpu().numpy() if boundary_contacts else None

        # Load panel
        panel_path_full = os.path.join(rootdir, projdir, panel_path)
        panel = pd.read_csv(panel_path_full)
        selected_markers = panel[panel['Full'] == 1].reset_index(drop=True)
        selected_indices = range(len(selected_markers)) 
        selected_names = selected_markers['Target'].values

        # Extract unique CellIDs
        cell_ids = torch.unique(cell_mask_tensor).cpu().numpy()

        # Prepare results
        results = []
        geom_results = []

        # Compartment data
        compartment_data = {}
        if compartment:
            for comp_name, comp_files in compartment_masks.items():
                comp_file_tif = comp_files.get(f"{name}{comp_suffix}.tif")
                comp_file_tiff = comp_files.get(f"{name}{comp_suffix}.tiff")
                comp_file = comp_file_tif if comp_file_tif else comp_file_tiff
                if comp_file:
                    compartment_data[comp_name] = torch.tensor(
                        imread(comp_file),
                        device=device, 
                        dtype=torch.float32
                    )

        # ----------------------------------------
        # Process each cell
        # ----------------------------------------
        for cell_id in cell_ids:
            cell_region = (cell_mask_tensor == cell_id)
            cell_data = {
                "Image": name,
                "CellID": int(cell_id)
            }

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

                cell_data["X"] = centroid_x
                cell_data["Y"] = centroid_y

                # Eccentricity
                bbox_indices = torch.nonzero(cell_region)
                if bbox_indices.shape[0] > 0:
                    height = bbox_indices[:, 0].max().item() - bbox_indices[:, 0].min().item() + 1
                    width  = bbox_indices[:, 1].max().item() - bbox_indices[:, 1].min().item() + 1
                    cell_data["Eccentricity"] = height / width if width != 0 else np.nan
                else:
                    cell_data["Eccentricity"] = np.nan

                # --------------------------------------------------
                # BOUNDARY CONTACT CALCULATIONS (Renamed & Combined)
                # --------------------------------------------------
                if boundary_contacts and cell_id != 0:
                    # Identify boundary pixels
                    cell_region_cpu = (cell_mask_cpu == cell_id)
                    structure = generate_binary_structure(2, 1)
                    dilated   = binary_dilation(cell_region_cpu, structure=structure)
                    boundary  = np.logical_xor(cell_region_cpu, dilated)
                    boundary_coords = np.argwhere(boundary)

                    # Keep a breakdown of how many boundary pixels touch each neighbour ID
                    neighbour_counts = defaultdict(int)

                    # Loop over each boundary pixel
                    for (yy, xx) in boundary_coords:
                        # Gather all distinct neighbour IDs (besides the cell itself)
                        neighbours_found = set()
                        # Check 8 neighbours
                        for ny in [yy-1, yy, yy+1]:
                            for nx in [xx-1, xx, xx+1]:
                                if (ny, nx) != (yy, xx):
                                    if (0 <= ny < cell_mask_cpu.shape[0]) and (0 <= nx < cell_mask_cpu.shape[1]):
                                        neighbour_id = cell_mask_cpu[ny, nx]
                                        if neighbour_id != cell_id:  
                                            neighbours_found.add(neighbour_id)
                        
                        # If we found any other cell(s), increment for each
                        if len(neighbours_found) > 0:
                            for n in neighbours_found:
                                neighbour_counts[n] += 1
                        else:
                            # If no other cell was found, we store "0" or something to indicate no contact
                            neighbour_counts[0] += 1

                    # Build an underscore-separated string like "2x20_3x13_0x40"
                    # Sort by neighbour ID for consistency
                    sorted_neighbours = sorted(neighbour_counts.items(), key=lambda x: x[0])
                    detail_str = "_".join(f"{int(nid)}x{count}" for (nid, count) in sorted_neighbours)
                    cell_data["boundary_contacts"] = detail_str

            # Mean intensity
            if mean:
                for idx, marker in zip(selected_indices, selected_names):
                    fluorescence_slice = fluorescence_stack_tensor[idx]
                    mean_intensity = torch.mean(fluorescence_slice[cell_region]).item()
                    cell_data[marker] = mean_intensity

            # Compartment measurement
            if compartment:
                for comp_name, comp_mask in compartment_data.items():
                    if compartment_measure == "centroid":
                        if not np.isnan(centroid_y) and not np.isnan(centroid_x):
                            comp_value = comp_mask[int(centroid_y), int(centroid_x)].item()
                        else:
                            comp_value = np.nan
                    elif compartment_measure == "mean":
                        comp_value = (torch.mean(comp_mask[cell_region]).item() 
                                      if cell_data["Area"] > 0 else np.nan)
                    elif compartment_measure == "mode":
                        values, counts = torch.unique(comp_mask[cell_region], return_counts=True)
                        comp_value = (
                            values[torch.argmax(counts)].item() 
                            if counts.numel() > 0 else np.nan
                        )
                    else:
                        raise ValueError(f"Invalid compartment_measure: {compartment_measure}")
                    cell_data[comp_name] = comp_value

            # Geometry
            if geometry:
                cell_region_cpu = cell_region.cpu().numpy()
                contours = find_contours(cell_region_cpu, level=0.5)
                geom = [contour.tolist() for contour in contours]
                geom_results.append({"Image": name, "CellID": int(cell_id), "Geometry": geom})

            # Store
            results.append(cell_data)

        # ---------------------------------------------------------
        # NEAREST NEIGHBOURS
        # ---------------------------------------------------------
        if neighbours > 0 and shape:
            # Build arrays for X, Y, CellID
            coords = []
            cellid_array = []
            for r in results:
                coords.append((r["Y"], r["X"]))   # (row, col)
                cellid_array.append(r["CellID"]) 
            
            coords = np.array(coords)
            cellid_array = np.array(cellid_array)

            # If we have at least 2 cells with valid coords
            if len(coords) > 1 and not np.isnan(coords).any():
                dist_matrix = cdist(coords, coords, metric="euclidean")

                for i, row_dist in enumerate(dist_matrix):
                    # i = index of the current cell
                    # sort by ascending distance
                    sorted_ix = np.argsort(row_dist)

                    # Remove the index to itself
                    sorted_ix = sorted_ix[sorted_ix != i]

                    # Filter out neighbours that are cellID 0
                    valid_ix = [ix for ix in sorted_ix if cellid_array[ix] != 0]

                    # Now pick up to K nearest from valid_ix
                    k = min(neighbours, len(valid_ix))
                    nearest_ids = cellid_array[valid_ix[:k]]
                    nearest_dists = row_dist[valid_ix[:k]]

                    # Convert to underscore-separated
                    nn_str   = "_".join(str(int(nid)) for nid in nearest_ids)
                    dist_str = "_".join(f"{d:.2f}" for d in nearest_dists)

                    results[i]["nearest_neighbours"] = nn_str
                    results[i]["nearest_neighbours_dist"] = dist_str
            else:
                # If there's only one cell or coords are invalid:
                for r in results:
                    r["nearest_neighbours"] = ""
                    r["nearest_neighbours_dist"] = ""

        # Collect results
        all_results.extend(results)
        all_geom_results.extend(geom_results)

    # ----------------------------------------------------------
    # Build DataFrames and output
    # ----------------------------------------------------------
    final_df = pd.DataFrame(all_results)
    geom_df  = pd.DataFrame(all_geom_results)

    output_path = os.path.join(rootdir, projdir, out_dir)
    geom_output_path = os.path.join(rootdir, projdir, geom_out_dir)

    final_df.to_csv(output_path, index=False)
    geom_df.to_csv(geom_output_path, index=False)

    print("Processing complete.")
    print("Total time taken:", time.time() - start_time)
    print(f"Results saved to {output_path}")
    print(f"Geometry results saved to {geom_output_path}")

