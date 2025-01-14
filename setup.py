from setuptools import setup, find_packages

setup(
    name="PyMComplete",
    version="0.0.1",
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),  
    description="The functions for the IMComplete preprocessing workflow are wrapped up in this small package called PyMComplete",
    author="Thomas O'Neil",
    author_email="thomas.oneil@sydney.edu.au",
    url="https://github.com/CVR-MucosalImmunology/IMComplete-Workflow",  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)