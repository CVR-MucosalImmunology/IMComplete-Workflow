import subprocess
import sys
import pkg_resources
import torch

def check_setup():
    """
    Checks for required Python packages and verifies CUDA installation.

    Returns:
        dict: A dictionary of missing or insufficient packages with their required versions.
    """
    print("Checking required packages in the current Conda environment...\n")
    
    required_packages = {
        "imcsegpipe": "1.0.0",
        "readimc": "0.8.0",
        "imc-denoise": "0.0.0",
       # "test1":"",
       # "test":"0",
    }

    missing_packages = {}

    for package, version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if version and pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                missing_packages[package] = version
        except pkg_resources.DistributionNotFound:
            missing_packages[package] = version

    if missing_packages:
        print("The following packages are missing or have insufficient versions:")
        for package, version in missing_packages.items():
            if version:
                print(f" - {package} (required version: {version})")
            else:
                print(f" - {package} (no version specified)")
    else:
        print("  All required packages are installed and meet the required versions.")

    print("\n-----------------\n\nChecking that CUDA has been installed properly...\n")
    if torch.cuda.is_available():
        print("  GPU acceleration via CUDA is available")
    else:
        print("  GPU acceleration has not been prepared. Consult https://pytorch.org/get-started/previous-versions/\nand try again")


