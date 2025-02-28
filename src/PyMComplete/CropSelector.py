import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.widgets import RectangleSelector 

def CropSelector(
        rootdir:str, 
        projdir:str,
        panel_path = "panel.csv",
        image_path = "image.csv",
        images_dir = "analysis/2_cleaned",
        suffix  = "_cleaned"):
    
    project_path = Path(rootdir) / projdir
    im_dir =  project_path / images_dir
    panel_path = project_path / panel_path
    sample_csv_path = project_path / image_path

    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    panel = pd.read_csv(panel_path)

    if not os.path.exists(sample_csv_path):
        raise FileNotFoundError(f"Image file not found: {sample_csv_path}")
    samples = pd.read_csv(sample_csv_path)

    if "Image" not in samples.columns:
        raise ValueError("image.csv must contain an 'Image' column.")
    if "Target" not in panel.columns:
        raise ValueError("panel.csv must contain a 'Target' column for channel names.")

    # We'll use the "Target" column as channel names
    channel_names = panel.loc[panel['Full'] == 1, 'Target'].tolist()
    num_channels  = len(channel_names)

    # Subfolders in 1_image_out
    # Check if im_dir contains subdirectories or TIFF files directly
    subdirs = [d for d in os.listdir(im_dir) if os.path.isdir(os.path.join(im_dir, d))]
    tiff_files = []

    if subdirs:
        # If there are subdirectories, list TIFF files within each subdirectory
        for subdir in subdirs:
            subdir_path = os.path.join(im_dir, subdir)
            tiff_files.extend([
                os.path.join(subdir, f) for f in os.listdir(subdir_path)
                if f.endswith(".tif") or f.endswith(".tiff")
            ])
    else:
        # If no subdirectories, list TIFF files directly in im_dir
        tiff_files = [
            f for f in os.listdir(im_dir)
                if f.endswith(".tif") or f.endswith(".tiff")
        ]

    tiff_files.sort()
    
    # 2. Create Interactive Widgets
    
    image_dropdown = widgets.Dropdown(
        options=tiff_files,
        description='Image:',
        layout=widgets.Layout(width='200px')
    )

    channel_dropdown = widgets.Dropdown(
        options=channel_names,
        description='Channel:',
        layout=widgets.Layout(width='200px')
    )

    crop_button = widgets.Button(
        description="Crop",
        button_style='success'
    )
    save_button = widgets.Button(
        description="Save",
        button_style='info'
    )

    output_display = widgets.Output()

    # We'll keep track of the figure, axes, selected ROI, and loaded data
    # in these variables. We'll define them in the function's closure so we can
    # access/update them in callbacks.
    fig       = None
    ax_left   = None
    ax_right  = None
    rect_sel  = None
    roi       = {"x": 0, "y": 0, "w": 0, "h": 0}  # will store rectangle coords
    current_channel_data = None      # the 2D channel image
    cropped_data         = None      # the cropped region

    #--------------------------------------------------------------------------
    # 3. RectangleSelector callback
    #--------------------------------------------------------------------------
    def on_select(eclick, erelease):
        """
        Called whenever the user finishes drawing or moving the rectangle.
        eclick/erelease: mouse events with xdata, ydata in axes coords
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure we have integer coords
        x_min, x_max = sorted([int(round(x1)), int(round(x2))])
        y_min, y_max = sorted([int(round(y1)), int(round(y2))])
        w = x_max - x_min
        h = y_max - y_min

        roi["x"], roi["y"], roi["w"], roi["h"] = x_min, y_min, w, h

    #--------------------------------------------------------------------------
    # 4. Function to load and display the selected channel
    #--------------------------------------------------------------------------
    def display_image(*args):
        """
        Loads the selected image & channel, sets up the RectangleSelector on the left axis.
        """
        nonlocal fig, ax_left, ax_right, rect_sel, current_channel_data, cropped_data
        cropped_data = None  # reset
        with output_display:
            clear_output(wait=True)

            selected_image_name = image_dropdown.value
            selected_channel_name = channel_dropdown.value

            # Convert that to a channel index
            try:
                channel_idx = channel_names.index(selected_channel_name)
            except ValueError:
                print(f"Channel '{selected_channel_name}' not found in panel.")
                return

            # Build the path to the TIF file (expected one TIF in the subfolder)
            tif_path = os.path.join(im_dir, selected_image_name)


            # Read the stack
            stack = tiff.imread(tif_path)
            if stack.shape[0] != num_channels:
                print(
                    f"Warning: panel has {num_channels} channels, "
                    f"but TIF has {stack.shape[0]} channels."
                )

            current_channel_data = stack[channel_idx, :, :]

            # Create a new figure
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
            fig.canvas.toolbar_visible = False  # optional: hide toolbar
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False

            # Display the channel on the left
            ax_left.imshow(current_channel_data, cmap='gray')
            ax_left.set_title(f"{selected_image_name} - {selected_channel_name}")
            ax_left.axis('off')

            # The right side is blank initially
            ax_right.imshow(np.zeros((10,10)), cmap='gray')
            ax_right.set_title("Cropped Region")
            ax_right.axis('off')

            # Create RectangleSelector for the left axis
            rect_sel = RectangleSelector(
                ax_left,
                onselect=on_select,        # your callback function
                useblit=False,
                interactive=True,          # let the user move/resize the rectangle
                button=[1],                # left mouse button
                props=dict(
                    facecolor='none',
                    edgecolor='red',
                    fill=False, alpha=1
                )
            )

            plt.tight_layout()
            plt.show()

    #--------------------------------------------------------------------------
    # Crop button callback
    #--------------------------------------------------------------------------
    def on_crop_clicked(b):
        """
        Uses the ROI (roi dict) to crop the currently displayed channel, then
        shows the result in ax_right.
        """
        nonlocal cropped_data

        if current_channel_data is None:
            print("No image loaded yet.")
            return

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if w <= 0 or h <= 0:
            print("Please draw a rectangle first.")
            return

        # Perform the crop
        cropped_data = current_channel_data[y:y+h, x:x+w]

        with output_display:
            clear_output(wait=False)  # keep the figure
            # We'll re-draw the figure, focusing on the right axis
            # The figure should already be defined
            ax_right.clear()
            ax_right.imshow(cropped_data, cmap='gray')
            ax_right.set_title(f"Cropped: x={x}, y={y}, w={w}, h={h}")
            ax_right.axis('off')
            plt.show()

    #--------------------------------------------------------------------------
    # Save button callback
    #--------------------------------------------------------------------------
    def on_save_clicked(b):
        """
        Saves the crop coords as "x_y_w_h" in image.csv for the row
        where samples['Image'] equals the selected image.
        """
        selected_image_name = image_dropdown.value
        imagename = os.path.splitext(selected_image_name)[0].replace(suffix, "")

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

        if w <= 0 or h <= 0:
            print("No valid crop selected. Did you draw a rectangle?")
            return
        coords_str = f"{x}_{y}_{w}_{h}_manual"

        # Find the row in samples where 'Image' = selected_image_name
        # If multiple rows match, will update all... 
        mask = (samples["Image"] == imagename)
        if not mask.any():
            print(f"No row in image.csv with Image == '{imagename}'.")
            return
        
        #Check if Crop exists as a column
        if "Crop" not in samples.columns:
            samples["Crop"] = np.nan
        samples["Crop"] = samples["Crop"].astype("string")

        samples.loc[mask, "Crop"] = coords_str

        # Save back to CSV
        samples.to_csv(sample_csv_path, index=False)
        print(f"Saved coords '{coords_str}' for image '{imagename}' into image.csv.")

    #--------------------------------------------------------------------------
    # 7. Wire up callbacks
    #--------------------------------------------------------------------------
    # Show or re-show the image whenever the user changes either dropdown
    image_dropdown.observe(display_image, names='value')
    channel_dropdown.observe(display_image, names='value')

    # Or whenever the function first runs
    # we can do an initial display after the UI is built.

    crop_button.on_click(on_crop_clicked)
    save_button.on_click(on_save_clicked)

    #--------------------------------------------------------------------------
    # 8. Layout the UI
    #--------------------------------------------------------------------------
    ui = widgets.VBox([
        widgets.HBox([image_dropdown, channel_dropdown]),
        widgets.HBox([crop_button, save_button]),
        output_display
    ])
    
    display(ui)

    # Trigger an initial display
    display_image()

