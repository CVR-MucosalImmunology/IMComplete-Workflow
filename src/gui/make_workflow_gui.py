
import os
import json
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def make_workflow_gui(dir):
    # Define module paths
    modules = {
        "intro": [
            {"type": "markdown", "path": "intro.txt"}
        ],
        "dirs": [
            {"type": "markdown", "path": "dirs_md.txt"}
        ],
        "checks": [
            {"type": "markdown", "path": "check_md.txt"},
            {"type": "code", "path": "check_code.txt"}
        ],
        "newProj": [
            {"type": "markdown", "path": "newproj_md.txt"},
            {"type": "code", "path": "newproj_code.txt"}
        ],
        "imc": [
            {"type": "markdown", "path": "imc_md.txt"},
            {"type": "code", "path": "imc_code.txt"}
        ],
        "image": [
            {"type": "markdown", "path": "image_md.txt"},
            {"type": "code", "path": "image_code.txt"}
        ]
    }

    # Prepare the notebook template
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "IMComplete",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.21"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Function to add a cell
    def add_cell(cell_type, content):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": content
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        notebook["cells"].append(cell)

    # Helper function to load module files
    def add_module(module_files):
        for file in module_files:
            if os.path.exists(file["path"]):
                with open(file["path"], "r") as f:
                    add_cell(file["type"], f.read())

    # GUI to select options
    def open_dialog():
        app = ttk.Window(themename="cosmo")  
        app.geometry("600x700") 
        app.title("Select Workflow Options")


        label = ttk.Label(app, text="Information") 
        label.pack(pady=30) 
        label.config(font=("Arial", 24, "bold")) 

        howto = ttk.Label(app, text = "Customise your own image processing workflow by selecting the relevant modules.\n\nSee [LINK] for examples of each module.")
        howto.pack(pady=10) 
        howto.config(font=("Arial", 14), justify="center") 

        ttk.Separator(bootstyle="info").pack(fill="x", pady=10)

               # Variable to store the selected directory and project name
        selected_directory = tk.StringVar()
        project_directory = tk.StringVar()
        full_dir = tk.StringVar(value="")  # Initialize as an empty string

        # Function to update the full directory
        def update_full_dir():
            dir_value = selected_directory.get()
            proj_value = project_directory.get()
            if dir_value and proj_value:
                full_dir.set(f"{dir_value}/{proj_value}")
            elif dir_value:
                full_dir.set(dir_value)
            else:
                full_dir.set("")
                        # Function to open the directory selection dialog
        def select_directory():
            directory = filedialog.askdirectory(title="Select Directory")
            if directory:
                selected_directory.set(directory)
                print(f"Selected Directory: {directory}")
                update_full_dir() 
        # Frame for project directory input
        projdirect = ttk.Frame(app) 
        projdirect.pack(pady=15, padx=10, fill="x")
        ttk.Label(projdirect, text="Project Name: ", anchor="w").pack(side="left", padx=5)
        ttk.Entry(projdirect, textvariable=project_directory).pack(side="right", padx=5) 

        # Frame for directory selection
        dir_button = ttk.Frame(app) 
        dir_button.pack(pady=15, padx=10, fill="x")        
        ttk.Button(
            dir_button, 
            text="Select Directory", 
            command=select_directory, 
            bootstyle="secondary"
        ).pack(side="left", padx=5)
        ttk.Label(dir_button, textvariable=full_dir, anchor="w").pack(side="right", fill="x", expand=True, padx=5)

        ttk.Separator(bootstyle="info").pack(fill="x", pady=10)

        # Variables for checkboxes
        options = {
            "default": ttk.BooleanVar(value=False),
            "checks_sel": ttk.BooleanVar(value=True),
            "newProj_sel": ttk.BooleanVar(value=True),
            "mcd_sel": ttk.BooleanVar(value=True)
        }

        default = ttk.Frame(app) 
        default.pack(pady=15, padx=10, fill="x")
        ttk.Label(default, text="Output all modules:", anchor="w").pack(side="left",fill="x", padx=5, expand=True)
        ttk.Checkbutton(default, variable=options["default"],bootstyle="round-toggle").pack(side="right", padx=5) 

        ttk.Separator(bootstyle="info").pack(fill="x", pady=10)

        sup = ttk.Frame(app) 
        sup.pack(pady=5, padx=10, fill="x")
        ttk.Label(sup, text="Check Setup Function (recommended): ", anchor="w").pack(side="left",fill="x", padx=5, expand=True)
        ttk.Checkbutton(sup, variable=options["checks_sel"],bootstyle="round-toggle").pack(side="right",padx=5) 

        nproj = ttk.Frame(app) 
        nproj.pack(pady=5, padx=10, fill="x")
        ttk.Label(nproj, text="Create New Project Function (recommended): ", anchor="w").pack(side="left",fill="x", padx=5, expand=True)
        ttk.Checkbutton(nproj, variable=options["newProj_sel"],bootstyle="round-toggle").pack(side="right", padx=5) 

        mcd = ttk.Frame(app) 
        mcd.pack(pady=5, padx=10, fill="x")
        ttk.Label(mcd, text="Image Format is MCD (e.g. IMC): ", anchor="w").pack(side="left", fill="x",padx=5, expand=True)
        ttk.Checkbutton(mcd, variable=options["mcd_sel"],bootstyle="round-toggle").pack(side="right",padx=5) 

        # Function to submit
        def submit():
            rootdir = selected_directory.get()
            projdir = project_directory.get()

            # Ensure rootdir and projdir are valid
            if not rootdir or not projdir:
                print("Error: Both directory and project name must be provided.")
                return
            # Create the project directory
            project_path = os.path.join(rootdir, projdir)
            os.makedirs(project_path, exist_ok=True)

            selected = {key: var.get() for key, var in options.items()}

            app.destroy()
            process_selection(selected, rootdir, projdir, project_path)

        # Add a submit button
        ttk.Button(app, text="Submit", command=submit, bootstyle="success").pack(side="left", fill="x", expand=True, padx=5) 

        app.mainloop()

    # Process user selections
    def process_selection(selected,rootdir, projdir, project_path):
        dirs_code = f"""
rootdir = '{rootdir}'
projdir = '{projdir}'
"""
        if selected["default"]:
            add_module(modules["intro"])
            add_module(modules["checks"])
            add_module(modules["dirs"])
            add_cell("code", dirs_code)
            add_module(modules["newProj"])
            add_module(modules["imc"])
            add_module(modules["image"])
        else:
            add_module(modules["intro"])
            if selected["checks_sel"]:
                add_module(modules["checks"])
            add_module(modules["dirs"])
            add_cell("code", dirs_code)

            if selected["newProj_sel"]:
                add_module(modules["newProj"])
            if selected["mcd_sel"]:
                add_module(modules["imc"])
            if not selected["mcd_sel"]:
                add_module(modules["image"])

        # Output the notebook as JSON
        output_file = os.path.join(project_path, "CustomisedWorkflow.ipynb")
        with open(output_file, "w") as f:
            json.dump(notebook, f, indent=4)

        print(f"Notebook generated at {output_file}")

    # Open dialog to get user input
    open_dialog()

make_workflow_gui(os.path.abspath(os.path.join(script_dir, "..", "..", "..")))
# Example usage:
# python IMComplete-Workflow/src/gui/run_generate_notebook.py
