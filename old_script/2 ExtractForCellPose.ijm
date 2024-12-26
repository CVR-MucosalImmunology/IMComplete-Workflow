/*
 * This macro creates 2-channel stack of nuclei and cell-body for segmentation using Cellpose. 
 * The cell-body image is created by normalising non-nuclei channels (values range between 0 - (2^16-1) = 65535) 
 * which were set to '1' in the 'Segment' column of "panel.csv".
 * 
 * Inputs: 
 * 	1. Image stacks in '1b_for_segmentation' folder
 * 	2. 'panel.csv' file
 * Outputs: 
 * 	1. 2-channel stack of nuclei and cell-body
 * 	2. Random cropped area of each image for training in Cellpose
 * 
 * NOTE: You need to set variables and directories in the first section below before running the script.
 * 
 * Authors: heeva.baharlou@gmail.com and oscardong4@gmail.com (29/04/2024)
 */
 
roiManager("reset");
run("Clear Results");
run("Close All");

setBatchMode(false);
run("Set Measurements...", "area mean min redirect=None decimal=3");


// **** 
// CHANGE: Set variables and directories 
// Set your 'analysis' folder directory
dir = "";
// Set your 'panel.csv' file directory
panelDir = "";
// Change this to what you call your nuclei stain under the 'Target' column in 'panel.csv'. This usually corresponds to the Ir191 or Ir193 Metal tags. 
dna = "DNA";
// Size (in pixels) of the square for cropping
square_size = 200;
// ****


// NO NEED TO CHANGE anything from here
dirImages = dir + "/1b_for_segmentation";
imOutput = dir +  "/2a_cellpose_full";
cropOutput =  dir + "/2b_cropped_images";

// Create the output directories if they don't exist
if(File.exists(imOutput) == 0) File.makeDirectory(imOutput);
if(File.exists(cropOutput) == 0) File.makeDirectory(cropOutput);

// Get list of images in the directory.
list = getFileList(dirImages);
a = newArray();
index = 0;
for (i=0; i<list.length; i++) {
    if (endsWith(list[i], ".tiff") || endsWith(list[i], ".png") || endsWith(list[i], ".tif")) {
        a[index] = dirImages + "/" + list[i];
        index = index + 1;
    }
}

list = a;

// Make an array of 'Targets' in panel.csv that were selected for segmentation ("1" in the 'Segment' column)
open(panelDir);

IJ.renameResults("panel.csv", "Results");
vMarkers = newArray();
index = 0;
for (i=0; i<nResults; i++) {
    ilastikValue = getResult("Segment", i);
    if (ilastikValue == 1) {
        a = getResultString("Target",i);
        vMarkers[index] = a;
        //save position of the nuclei stain in the array. Use later to select this channel. 
        if(a == dna) dnaIndex = index;
        index = index + 1;
    }
}

// Go through each image and add to the manager if not to be excluded
for (i=0; i<list.length; i++) {
    
    open(list[i]);
    imTitle = getTitle();
    imTitle = substring(imTitle, 0, lengthOf(imTitle)-5);
    
	// Normalise all images from stack so values span 0 - (2^16 -1)
	stackTitle = getTitle();
	run("Split Channels");
	imageList = getList("image.titles");
	for (j = 0; j < lengthOf(imageList); j++) {
	    selectWindow(imageList[j]);
	    run("Enhance Contrast...", "saturated=0.10 normalize");
	    //rename images by channel number so you can select the DNA channel out later. 
	    rename(toString(j));
	    }
	
	// Re-stack images excluding nuclei channel

	selectWindow(dnaIndex);
	//add slice so that the dna channel doesn't get stacked
	run("Add Slice");
	run("Images to Stack", "use keep");
	run("Z Project...", "projection=[Average Intensity]");
	body = getTitle();
	
	// Add duplicate channel 
	selectWindow(dnaIndex);
	run("Duplicate...", "duplicate");
	rename("empty");
	Stack.getDimensions(width, height, channels, slices, frames);
	run("Delete Slice");
	for(x=0; x<width; x++) {
		for(y=0;y<height;y++){
			setPixel(x,y,0);
		}
	}

	// Remove slice from dna channel and make stack with projected image (cell body)
	selectWindow(dnaIndex);
	run("Delete Slice");
	run("Merge Channels...", "c1=empty c2=AVG_Stack c3=" + dnaIndex + " create");
	run("Stack to Hyperstack...", "order=xyczt(default) channels=3 slices=1 frames=1 display=Composite");
	
	// Adjust contrast
	Stack.setDisplayMode("color");
	Stack.setChannel(1);
	run("Red");

	Stack.setChannel(3);
	run("Blue");
	run("Clear Results");
	run("Measure");
	setMinAndMax(0, getResult("Max", 0)/2);

	Stack.setChannel(2);
	run("Green");
	run("Clear Results");
	run("Measure");
	setMinAndMax(0, getResult("Max", 0)/2);

	Stack.setDisplayMode("composite");
	
	// Name image and save
	rename(imTitle + "_CpSeg");
	saveAs("tiff", imOutput + "/" + getTitle());
	
	// Crop random area and save
	Stack.getDimensions(width, height, channels, slices, frames);
	// Create area so that square isn't outside the image
	workable_x = width-square_size;
	workable_y = height-square_size;
	rand_x = random*workable_x;
	rand_y = random*workable_y;
	makeRectangle(rand_x, rand_y, square_size, square_size);
	run("Duplicate...", "duplicate");
	rename(imTitle + "_CpCrop");
	saveAs("tiff", cropOutput + "/" + getTitle());
	
	roiManager("reset");
	run("Clear Results");
	run("Close All");

}
