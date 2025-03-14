import os
import skimage.io
from cellpose import models
from cellpose.io import logger_setup
from pathlib import Path
import tifffile

def BatchSegment(
        rootdir,
        projdir,
        model = None, 
        builtin_model = None, 
        channels = [2, 3],
        cell_diameter: int,
        flow_threshold: int,
        cellprob_threshold: int,
        model_dir ="analysis/3_segmentation/3a_cellpose_crop/models",
        full_from = "analysis/3_segmentation/3b_cellpose_full",
        mask_to = "analysis/3_segmentation/3c_cellpose_mask",
        suffix = "_mask"
        ):
    
    # Define Cellpose model
    if model is not None: 
        model_path = os.path.join(rootdir, projdir,model_dir, model)
        if os.path.exists(model_path):
            print("Choosing ", model_path)
            model = models.CellposeModel(pretrained_model=model_path)

        else:
            print("Model path does not exist. Exiting...")
            print(model_path)
            return
        
    elif model is None and builtin_model is not None: 
        if builtin_model in ['cyto3', 'cyto2', 'cyto', 'nuclei']:
            print("Choosing ", builtin_model)
            model = models.Cellpose(model_type=builtin_model)
        elif builtin_model in ['tissuenet_cp3', 'livecell_cp3', 'yeast_PhC_cp3','yeast_BF_cp3', 'bact_phase_cp3', 'bact_fluor_cp3', 'deepbacs_cp3', 'cyto2_cp3']:
            model=models.CellposeModel(model_type='tissuenet_cp3')
        else: 
            print("'",builtin_model, "' not available as a built in model.")
            print("Choose: cyto, cyto2, cyto3, nuclei, tissuenet_cp3, livecell_cp3, yeast_PhC_cp3,yeast_BF_cp3, bact_phase_cp3, bact_fluor_cp3, deepbacs_cp3, or cyto2_cp3.")
            return

    # Set and create directories
    analysis = Path(os.path.join(rootdir, projdir))
    image_dir = analysis / full_from
    mask_dir = analysis / mask_to

    # Call logger_setup to have output of cellpose written
    logger_setup()

    # Get list of image files
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tiff")] 
    imgs = [tifffile.imread(f) for f in files]

    # Run segmentation
    masks, flows, styles  = model.eval(imgs, diameter=cell_diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, channels=channels)

    # Save mask images
    for idx, mask in enumerate(masks):
        original_path = Path(files[idx])
        new_path = mask_dir / (original_path.stem + suffix +".tif")
        skimage.io.imsave(new_path, mask)

    print("Done!")