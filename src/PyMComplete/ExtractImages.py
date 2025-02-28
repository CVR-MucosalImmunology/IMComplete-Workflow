from pathlib import Path
import pandas as pd
from skimage import io

def ExtractImages(rootdir:str,
                   projdir:str,
                   format:str,
                   rawimage_dir = "raw",
                   extract_dir = "analysis/1_image_out"):
    
    project_path = Path(rootdir) / projdir
    images_dir = project_path / extract_dir
    raw = project_path / rawimage_dir
    
    if format == "if":
        print("Extracting Immunofluorescent Images...\n")
        for sample_dir in raw.iterdir():
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue

            # Create subfolder in analysis/1_image_out
            acquisition_subdir = images_dir / sample_dir.name
            acquisition_subdir.mkdir(parents=True, exist_ok=True)

            # Final stacked TIFF path
            out_tiff_path = acquisition_subdir / f"{sample_dir.name}.tiff"

            # We expect exactly one TIF in the folder
            tif_files = list(sample_dir.glob("*.tif*"))
            if len(tif_files) != 1:
                raise ValueError(
                    f"Expected exactly 1 TIF in '{sample_dir.name}', found {len(tif_files)}."
                )
            single_tif = tif_files[0]

            # Read the stack
            image = io.imread(str(single_tif))

            # Validate that the stack depth == number of rows in panel.csv
            if image.shape[0] != len(panel):
                raise ValueError(
                    f"Panel length is {len(panel)} but found `{image.shape[0]}` channels"
                    f" in '{sample_dir.name}'."
                )

            # Save the original stack (unprocessed or raw)
            io.imsave(str(out_tiff_path), image)
        print("Done!\n")

    if format == "imc":
        print("Extracting IMC images using Bodenmiller's extract_zip_file function ...\n")

        from tempfile import TemporaryDirectory
        import imcsegpipe
                
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

        print("Done!")

    # Create image.csv
    image_data = []
    for subdir in images_dir.iterdir():
        if subdir.is_dir():
            for tiff_file in subdir.glob("*.tiff"):
                image_data.append({
                    "Image": tiff_file.stem,
                    "ImShort": "",  # Example short name, adjust as needed
                    "ROI": "",  # Placeholder, adjust as needed
                    "ImageID": "",  # Example ID, adjust as needed
                    "DonorID": "",  # Placeholder, adjust as needed
                    "Condition": "",  # Placeholder, adjust as needed
                    "Crop": ""  # Placeholder, adjust as needed
                })

    image_df = pd.DataFrame(image_data)
    image_csv_path = images_dir / "image.csv"
    image_df.to_csv(image_csv_path, index=False)
    print(f"image.csv created at {image_csv_path}")