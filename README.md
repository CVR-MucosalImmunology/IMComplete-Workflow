# **Part A: Preprocessing of imaging data**  
<i>**Latest update</i> - Jan 2025**  

#### **Authors:**  
[Thomas O'Neil](https://github.com/DrThomasOneil) (thomas.oneil@sydney.edu.au) | [Oscar Dong](https://github.com/Awesomenous) (oscardong4@gmail.com) | [Heeva Baharlou](heeva.baharlou@sydney.edu.com)  

##### The purpose of this notebook is to provide a consolidated approach to IMC analysis and forms the prerequisite steps to the IMComplete R package workflow. We focused 

Nature Method of the Year in 2024 was [**spatial proteomics**](https://www.nature.com/articles/s41592-024-02565-3). 

> Computational tools for spatial proteomics are the focus of the second Comment, from Yuval Bussi and Leeat Keren. These authors note that current image processing and analysis workflow are **well defined but fragmented**, with various steps happening back to back **rather than in an integrated fashion**. They envision a future for the field where **image processing and analysis steps work in concert** for improved biological discovery.

In alignment with these comments, we have committed to provide a comprehensive and dynamic workflow. In part, we aimed to achieve this by compiling as much as we could into this pre-processing workflow. 

Particularly, we have emphasised tools that can be performed in <strong>*one*</strong> linear workflow. For example, we provide the function `PyProfiler`, a tool that performs the same functions as CellProfiler in extracting cell features, and RegisterImages to register IMC to IF in Python, and allowing users remain in this linear pipeline and not have to install additional applications.

<hr>

Some scripts adapted from [BodenmillerGroup/ImcSegmentationPipeline](https://github.com/BodenmillerGroup/ImcSegmentationPipeline)

<i>**Therefore, make sure to also reference these studies:**</i>  
- Windhager, J., Zanotelli, V.R.T., Schulz, D. et al. An end-to-end workflow for multiplexed image processing and analysis. [Nat Protoc](https://doi.org/10.1038/s41596-023-00881-0) (2023).  


<br>
<hr>


## Folder structure

```text
ImagingAnalysis/ (root directory)
├── IMComplete-Workflow
├── ImcSegmentationPipeline
├── Experiment_name_1
│     └── raw
│            └── Sample1.zip
│            └── Sample2.zip
│            └── ...
│     └── analysis
│            └── 1_image_out
│            └── 2_cleaned
│            └── 3_segmentation
│                   └── 3a_cellpose_crop
│                   └── 3b_cellpose_full
│                   └── 3c_cellpose_mask
│                   └── 3d_compartments
│            └── 4_pyprofiler_output
│     └── panel.csv
├── ...
├── Experiment_name_n
```
<br>
<hr> <hr>

# Set up

Anaconda is needed to run this workflow. Follow the steps below to set up Anaconda and a `conda` environment:

Install [**Anaconda** ](https://www.anaconda.com/download) and navigate to the relevant command line interface:
<br>
<div align="left">

| Windows                                                                                            | macOS                                                                                                      |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 1. Search for **'Anaconda Prompt'** in the taskbar search <br> 2. Select **Anaconda Prompt**  <br> | 1. Use `cmd + space` to open Spotlight Search  <br> 2. Type **'Terminal'** and press `return` to open <br> |

</div>
<br>

<hr><hr>

### *Using Anaconda...*

#### **Step 1:** Set your directory to the analysis folder (or the `root directory` for image analysis)

```bash
cd /Desktop/ImageAnalysis
```
<hr>

#### **Step 2:** Clone the IMComplete repository.

<storng>*From Github*</strong>  
Go to the [Github page](https://github.com/CVR-MucosalImmunology/IMComplete-Workflow) and near the top click the `code` button and download the zip. Unzip the folder into the `root` directory. This will contain the IMComplete-Workflow documents and allow ready access to the necessary files.

</strong>*Using Git*</strong> in command line

<details><summary>Install Git</summary>

Git needs to be installed on your system. Find the instructions [here](https://git-scm.com/downloads)

<hr></details>

```bash
git clone --recursive https://github.com/CVR-MucosalImmunology/IMComplete-Workflow.git
``` 
<hr>

#### **Step 3:** Clone the extra repositories: 

- [BodenmillerGroup/ImcSegmentationPipeline](https://github.com/BodenmillerGroup/ImcSegmentationPipeline): Windhager, J., Zanotelli, V.R.T., Schulz, D. et al. An end-to-end workflow for multiplexed image processing and analysis. [Nat Protoc](https://doi.org/10.1038/s41596-023-00881-0) (2023).  

```bash
git clone --recursive https://github.com/BodenmillerGroup/ImcSegmentationPipeline.git
```
<!---  
- [deMirandaLab/PENGUIN](https://github.com/deMirandaLab/PENGUIN): Sequeira, A. M., Ijsselsteijn, M. E., Rocha, M., & de Miranda, N. F. (2024). PENGUIN: A rapid and efficient image preprocessing tool for multiplexed spatial proteomics. [Computational and Structural Biotechnology Journal](https://doi.org/10.1101/2024.07.01.601513)
```bash
git clone --recursive https://github.com/deMirandaLab/PENGUIN.git
```
<--->

<hr>

#### **Step 4:** Create a conda environment and install some  packages (in one line)

```bash
conda env create -f IMComplete-Workflow/environment.yml
```

*This can take some time so be patient!*

<hr>

#### **Step 5:** Activate the newly created conda environment

```bash
conda activate IMComplete
```

<hr>

#### **Step 6:** Activate and ensure your GPU-acceleration is accessible

Unfortunately, parts of this workflow will require GPU-acceleration: Cell segmentation, Denoise, PyProfiler (will run quicker, but not necessary).

You will need to install Pytorch and pytorch-cuda versions that are suitable for your PC. Instructions are found [here](https://pytorch.org/get-started/previous-versions/). The code will look like this:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

<hr>

#### **Step 7:** Select the IMComplete kernel in your IDE

If you are using VSCode, you'll see this option in the top right of the window. 

If you are using a jupyter notebook, you will see this...<span style="color:white; background:red">[TO ADD]</span>

<hr><hr>

## Workflow

1. Set up (`CheckSetup()`)  

2. Create a new project (`NewProject()`) 

3. Prep the raw folder and `panel.csv` 

4. Extract images from the raw folder (`ExtractImages()`) 

- *Optional 1:* Check filter parameters of IF data (`CheckExtract()`) 

- *Optional 2:* Filter images (`FilterImages()`) 

- *Optional 3:* Select crop regions for segmentation training (`CropSelector()`)  

5. Prepare the images for Segmentation model training (`PrepCellpose()`) 

- *Optional 4:* Register low-resolution images with high-resolution images to improve cell segmentation (`RegisterImages()`) 

6. Train a segmentation model (`cellpose`) 

- *Optional 5:* You have the option to not train a segmentation model and use a generic model.  

7. Batch segment the images and generate cell masks (`BatchSegment()`)

8. Extract data from your images using the cell segment masks (`PyProfiler()`)

<hr><hr>