from pathlib import Path
import pandas as pd
from skimage import io,  img_as_uint
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian

def FilterImages(rootdir,
                   projdir,
                   panel_filename="panel.csv",
                   format = "imc", 
                   hotpixel=None, 
                   gauss_blur=None, 
                   fullstack = True, 
                   hpf=50,
                   extract_dir = "analysis/1_image_out", 
                   clean_dir = "analysis/2_cleaned", 
                   suffix="_cleaned"):
    
    project_path = Path(rootdir) / projdir

    images_dir = project_path / extract_dir
    cleaned_dir = project_path / clean_dir
    panel_path = project_path / panel_filename

    panel = pd.read_csv(panel_path)

    if format == "if":
        def remove_hotpixels_threshold(img, threshold=5.0, neighborhood_size=3):
            """
            Replace 'hot' pixels that are above (threshold * local_mean) with that local mean.
            """
            img_float = img.astype(float)
            local_mean = uniform_filter(img_float, size=neighborhood_size)
            
            # create mask of hot pixels
            hot_mask = img_float > (threshold * local_mean)

            cleaned_img = img_float.copy()
            cleaned_img[hot_mask] = local_mean[hot_mask]

            # Convert back to original dtype (e.g., uint16) if desired
            return cleaned_img.astype(img.dtype)

        def apply_gaussian_blur(img, sigma=1.0):
                """
                Applies a Gaussian blur with a given sigma.
                Returns the blurred image (preserving the original range).
                """
                blurred = gaussian(img, sigma=sigma, preserve_range=True)
                return blurred.astype(img.dtype)
    
        for sample_dir in images_dir.iterdir():
            full_stack = []
            # Skip hidden folders (for whatever reason they may exist)
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue

            # We expect exactly one tiff. in the folder, check and then just read the first one.
            tif_files = list(sample_dir.glob("*.tif*"))
            if len(tif_files) != 1:
                raise ValueError(
                    f"Expected exactly 1 TIF in '{sample_dir.name}', found {len(tif_files)}."
                )
            single_tif = tif_files[0]
            image = io.imread(str(single_tif))

            # Check that the stack depth == number of rows in panel.csv
            if image.shape[0] != len(panel):
                raise ValueError(
                    f"Panel length is {len(panel)} but found `{image.shape[0]}` channels"
                    f" in '{sample_dir.name}'."
                )

            # Process each channel in the stack
            for idx in range(len(panel)):
                channel = image[idx, :, :]

                # 1) Hotpixel removal if threshold is not None
                if hotpixel and hotpixel.get("threshold") is not None:
                    channel = remove_hotpixels_threshold(
                        channel,
                        threshold=hotpixel["threshold"],
                        neighborhood_size=hotpixel.get("neighborhood", 3)
                    )

                # 2) Gaussian blur if gauss_blur is not None
                if gauss_blur is not None:
                    channel = apply_gaussian_blur(channel, sigma=gauss_blur)

                if panel.loc[idx, "Full"] == 1:
                    full_stack.append(img_as_uint(channel))
                # Replace the channel in the stack
                image[idx, :, :] = channel

            cleaned_image_name = f"{sample_dir.name}{suffix}.tiff"
            cleaned_image_path = cleaned_dir / cleaned_image_name
            
            if fullstack == True:
                if len(full_stack) > 0:
                    full_stack = np.stack(full_stack)
                    io.imsave(str(cleaned_image_path), full_stack)
                else:
                    print(f"Warning: No 'Full' channels found for sample '{sample_dir.name}'.")
            else:
                io.imsave(str(cleaned_image_path), image)



    elif format == "imc":    

        print("Generating Cleaned Images...\n")
        print("Using hot pixel filter of ",hpf,".\n")
        import imcsegpipe
        from imcsegpipe.utils import sort_channels_by_mass

        for image_dir in images_dir.glob("[!.]*"):
            if image_dir.is_dir():
                imcsegpipe.create_analysis_stacks(
                    acquisition_dir=image_dir,
                    analysis_dir=cleaned_dir,
                    analysis_channels=sort_channels_by_mass(
                        panel.loc[panel["Full"] == 1, "Conjugate"].tolist()
                    ),
                    suffix=suffix,
                    hpf=hpf
                )

    elif format != "if" and format != "imc":
        print("Format not specified. Choose 'imc' or 'if' specifically and run again.\n")
