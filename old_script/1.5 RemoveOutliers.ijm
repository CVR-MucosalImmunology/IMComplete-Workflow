/*
 * This macro removes bright objects from each of the IMC channels in your full images. 
 * Inputs: 
 * 	1. Image stacks in '1c_full_images' folder
 * Outputs: 
 * 	1. Image stacks with outliers removed
 * 
 * NOTE: You need to set variables and directories in the first section below before running the script.
 * 
 * Authors: oscardong4@gmail.com and heeva.baharlou@gmail.com (25/05/2024)
 */

setBatchMode(false);


// **** 
// CHANGE: Set variables and directories 
// Set your 'analysis' folder directory
dir = "";
// ****


// NO NEED TO CHANGE anything from here
fullPath = dir + "/1c_full_images"
// Get the list of files in image folder
files = getFileList(fullPath);

for (i = 0; i < files.length; i++) {
	// Process each .tiff image inside the image folder
	if (endsWith(files[i], ".tiff")) {
		open(fullPath + "/" + files[i]);
		// Remove outliers
		run("Remove Outliers...", "radius=5 threshold=50 which=Bright stack");
		// Prepare output path and save
		outPath = fullPath + "/" + files[i];
		saveAs("Tiff", outPath);
		close();
	}
}