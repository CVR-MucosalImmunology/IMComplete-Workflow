import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.widgets import RectangleSelector 

# --- Monkey Patch for Windows ipympl 0.9.3 ---
import matplotlib.backends.backend_webagg_core as webagg_core
if not hasattr(webagg_core.FigureCanvasWebAggCore, "_patched"):
    original_handle_event = webagg_core.FigureCanvasWebAggCore.handle_event
    def patched_handle_event(self, event):
        if 'buttons' not in event:
            event['buttons'] = 0  # add a default value if missing
        return original_handle_event(self, event)
    webagg_core.FigureCanvasWebAggCore.handle_event = patched_handle_event
    webagg_core.FigureCanvasWebAggCore._patched = True
# --- End of Monkey Patch ---

def CropSelector(
        rootdir: str, 
        projdir: str,
        panel_path="panel.csv",
        image_path="image.csv",
        images_dir="analysis/2_cleaned",
        suffix="_cleaned"):
    
    project_path = Path(rootdir) / projdir
    im_dir = project_path / images_dir
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

    # Use the "Target" column as channel names
    channel_names = panel.loc[panel['Full'] == 1, 'Target'].tolist()
    num_channels  = len(channel_names)

    # Determine TIFF files (in subdirectories or directly in the folder)
    subdirs = [d for d in os.listdir(im_dir) if os.path.isdir(os.path.join(im_dir, d))]
    tiff_files = []
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(im_dir, subdir)
            tiff_files.extend([
                os.path.join(subdir, f) for f in os.listdir(subdir_path)
                if f.endswith(".tif") or f.endswith(".tiff")
            ])
    else:
        tiff_files = [f for f in os.listdir(im_dir)
                      if f.endswith(".tif") or f.endswith(".tiff")]
    tiff_files.sort()
    
    # Create Interactive Widgets
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

    # Variables to store the state
    fig = None
    ax_left = None
    ax_right = None
    rect_sel = None
    roi = {"x": 0, "y": 0, "w": 0, "h": 0}  # store rectangle coordinates
    current_channel_data = None      # the 2D channel image
    cropped_data = None              # the cropped region

    # RectangleSelector callback (unchanged)
    def on_select(eclick, erelease):
        """
        Called when the user finishes drawing/moving the rectangle.
        """
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            print("Invalid selection. Please try again.")
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x_min, x_max = sorted([int(round(x1)), int(round(x2))])
        y_min, y_max = sorted([int(round(y1)), int(round(y2))])
        roi["x"], roi["y"], roi["w"], roi["h"] = x_min, y_min, x_max - x_min, y_max - y_min

    # Function to load and display the selected channel
    def display_image(*args):
        nonlocal fig, ax_left, ax_right, rect_sel, current_channel_data, cropped_data
        cropped_data = None  # reset previous crop
        if fig is not None:
            plt.close(fig)

        with output_display:
            clear_output(wait=True)

            selected_image_name = image_dropdown.value
            selected_channel_name = channel_dropdown.value

            try:
                channel_idx = channel_names.index(selected_channel_name)
            except ValueError:
                print(f"Channel '{selected_channel_name}' not found in panel.")
                return

            tif_path = os.path.join(im_dir, selected_image_name)
            stack = tiff.imread(tif_path)
            if stack.shape[0] != num_channels:
                print(
                    f"Warning: panel has {num_channels} channels, but TIF has {stack.shape[0]} channels."
                )
            current_channel_data = stack[channel_idx, :, :]

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
            if hasattr(fig.canvas, "toolbar_visible"):
                fig.canvas.toolbar_visible = False
                fig.canvas.header_visible = False
                fig.canvas.footer_visible = False

            ax_left.imshow(current_channel_data, cmap='gray')
            ax_left.set_title(f"{selected_image_name} - {selected_channel_name}")
            ax_left.axis('off')

            ax_right.imshow(np.zeros((10, 10)), cmap='gray')
            ax_right.set_title("Cropped Region")
            ax_right.axis('off')

            # The RectangleSelector is left exactly as you provided.
            rect_sel = RectangleSelector(
                ax_left,
                onselect=on_select,
                useblit=False,
                interactive=True,
                button=[1],
                props=dict(facecolor='none', edgecolor='red', fill=False, alpha=1)
            )

            plt.tight_layout()
            plt.show()

    # Crop button callback
    def on_crop_clicked(b):
        nonlocal cropped_data
        if current_channel_data is None:
            print("No image loaded yet.")
            return

        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if w <= 0 or h <= 0:
            print("Please draw a rectangle first.")
            return

        cropped_data = current_channel_data[y:y+h, x:x+w]
        with output_display:
            clear_output(wait=False)
            ax_right.clear()
            ax_right.imshow(cropped_data, cmap='gray')
            ax_right.set_title(f"Cropped: x={x}, y={y}, w={w}, h={h}")
            ax_right.axis('off')
            plt.show()

    # Save button callback
    def on_save_clicked(b):
        selected_image_name = image_dropdown.value
        imagename = os.path.splitext(selected_image_name)[0].replace(suffix, "")
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if w <= 0 or h <= 0:
            print("No valid crop selected. Did you draw a rectangle?")
            return
        coords_str = f"{x}_{y}_{w}_{h}_manual"
        mask = (samples["Image"] == imagename)
        if not mask.any():
            print(f"No row in image.csv with Image == '{imagename}'.")
            return
        if "Crop" not in samples.columns:
            samples["Crop"] = np.nan
        samples["Crop"] = samples["Crop"].astype("string")
        samples.loc[mask, "Crop"] = coords_str
        samples.to_csv(sample_csv_path, index=False)
        print(f"Saved coords '{coords_str}' for image '{imagename}' into image.csv.")

    # Wire up callbacks
    image_dropdown.observe(display_image, names='value')
    channel_dropdown.observe(display_image, names='value')
    crop_button.on_click(on_crop_clicked)
    save_button.on_click(on_save_clicked)

    ui = widgets.VBox([
        widgets.HBox([image_dropdown, channel_dropdown]),
        widgets.HBox([crop_button, save_button]),
        output_display
    ])
    
    display(ui)
    display_image()
