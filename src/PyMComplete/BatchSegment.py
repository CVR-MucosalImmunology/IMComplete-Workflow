import os
import pandas as pd
import skimage.io
from cellpose import models
from cellpose.io import logger_setup
from pathlib import Path
import tifffile

def BatchSegment(
        rootdir,
        projdir,
        model=None,
        builtin_model=None,
        channels=[2, 3],
        cell_diameter=14,
        flow_threshold=0.4,
        cellprob_threshold=0,
        model_dir="analysis/3_segmentation/3a_cellpose_crop/models",
        full_from="analysis/3_segmentation/3b_cellpose_full",
        mask_to="analysis/3_segmentation/3c_cellpose_mask",
        in_suffix="_CpSeg",
        out_suffix="_mask",
        image_csv="image.csv",
        image_col = "Image",
        extension=".tiff"
    ):
    # Define Cellpose model
    if model is not None:
        model_path = os.path.join(rootdir, projdir, model_dir, model)
        if os.path.exists(model_path):
            print("Choosing", model_path)
            model = models.CellposeModel(pretrained_model=model_path)
        else:
            print("Model path does not exist. Exiting...")
            print(model_path)
            return
    elif model is None and builtin_model is not None:
        if builtin_model in ['cyto3', 'cyto2', 'cyto', 'nuclei']:
            print("Choosing", builtin_model)
            model = models.Cellpose(model_type=builtin_model)
        elif builtin_model in ['tissuenet_cp3', 'livecell_cp3', 'yeast_PhC_cp3',
                               'yeast_BF_cp3', 'bact_phase_cp3', 'bact_fluor_cp3',
                               'deepbacs_cp3', 'cyto2_cp3']:
            model = models.CellposeModel(model_type='tissuenet_cp3')
        else:
            print(f"'{builtin_model}' not available as a built in model.")
            print("Choose: cyto, cyto2, cyto3, nuclei, tissuenet_cp3, livecell_cp3, yeast_PhC_cp3, yeast_BF_cp3, "
                  "bact_phase_cp3, bact_fluor_cp3, deepbacs_cp3, or cyto2_cp3.")
            return

    # Read CSV and check required column
    df_image = pd.read_csv(os.path.join(rootdir,projdir, image_csv))
    image_list = df_image[image_col].tolist()

    # Define directories
    analysis = Path(os.path.join(rootdir, projdir))
    image_dir = analysis / full_from
    mask_dir = analysis / mask_to
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Set up Cellpose logging
    logger_setup()

    # Construct file paths from the CSV names and input suffix.
    # Here we assume images are stored as .tif files.
    files = [image_dir / f"{img}{in_suffix}{extension}" for img in image_list]

    # Read images
    imgs = [tifffile.imread(str(f)) for f in files]

    # Run segmentation
    masks, flows, styles = model.eval(
        imgs,
        diameter=cell_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=channels
    )

    # Save mask images by replacing in_suffix with out_suffix in the file stem.
    for idx, mask in enumerate(masks):
        original_path = files[idx]
        new_stem = original_path.stem.replace(in_suffix, out_suffix)
        new_path = mask_dir / (new_stem + ".tif")
        skimage.io.imsave(str(new_path), mask)

    print("Done!")
