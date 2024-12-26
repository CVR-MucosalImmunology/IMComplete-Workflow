/*
 * This macro generates masks for epithelial areas in an image stack by extracting a user-specified channel, normalizing it, 
 * applying a user defined threshold, filtering out small particles, eroding the mask to remove residual attachments to the epithelium, 
 * and then dilating the mask. A delay is built into the macro to allow the user to visually check the results during processing.
 * 
 * Inputs:
 * 	Essential:
 * 		dir: path to your 'analysis' folder
 * 		channelNo: channel number containing your epithelial stain
 * 	Optional:
 * 		thresh: threshold (default is "Triangle")
 * 		epiSize: minimum pixel area size (default is 150)
 * 		blurr: Gaussian blur value (default is 0.75)
 * 		erodeNum: number of erosions (default is 2)
 * 		dilateNum: number of dilations (default is 2)
 * 		delayTime: delay time in milliseconds (default is 1000)
 * Outputs:
 * 	Epithelial mask images saved in your 'analysis/3b_for_cellprofiler' folder
 *
 * NOTE: You need to set variables and directories in the first section below before running the script.
 * 
 * Authors: heeva.baharlou@gmail.com and oscardong4@gmail.com (29/04/2024)
 */


// Set to 'true' to run in headless mode (faster and images don't pop up on screen)
setBatchMode(false);

roiManager("reset");
run("Clear Results");
run("Close All");


// **** 
// ESSENTIAL TO CHANGE: Set variables and directories 
// Set your 'analysis' folder directory
dir = "";
// Specify the channel number in the full image stack that contains your epithelial stain (eg. E-cadherin)
channelNo = 7;
// ****

// ####
// OPTIONAL TO CHANGE: Modifiable segmentation variables
// Apply threshold. Takes either an integer number up to 65535 or a preset algorithm like "Triangle".
thresh = "Triangle";
// After segmenting will remove epithelium below this pixel area size. Increase to remove more scraps.
epiSize = 150;
// Gaussian blur. Increase if too many grainy dots in final mask or want mask to look smoother.
blurr = 0.75;
// Erode x2 then dilate x2 to get rid of small attachments to larger epithelium. Increase if too many scraps emanating from epithelium you need to trim off. 
erodeNum = 2;
dilateNum = 2;
// Built-in delay so user can check if happy with result. Also prints number of current image for user to note down if any issues with image. 
// 1000 is 1 second. The image name then pops up for delayTime + 2 seconds to give you time to write down the image name.  
delayTime = 1000;
// ####


// NO NEED TO CHANGE anything from here
dirImages = dir + "/1c_full_images";
dirOutput = dir + "/3b_for_cellprofiler";
if(File.exists(dirOutput) == 0) File.makeDirectory(dirOutput);

// Get list of images in the directory
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

// Create an array of integers up to the number of images in the array 'a'
intArray = newArray(a.length);
for (i=0; i<a.length; i++) {
    intArray[i] = i + 1; // +1 because we want to start from 1 instead of 0
}
Array.show(intArray, a);

// Process each image
for (i=0; i<list.length; i++) {
    // Open the image
    open(list[i]);

    // Select the desired channel
    run("Duplicate...", "title=channel_"+channelNo+".tif duplicate channels="+channelNo);
	
	//Normalise scale of values. Helps with thresholding being consistent. 
	run("Enhance Contrast...", "saturated=0.35 normalize");
	
    // Apply Gaussian blur. Change Sigma as desired. Default is 0.75 which means 0.75 pixel blur. 
    run("Gaussian Blur...", "sigma=" + blurr + " stack");
	
	//Set Measurements to limit to threshold.
	run("Set Measurements...", "area mean min limit redirect=None decimal=3");
    
    // Apply threshold. Takes either an integer number up to 65535 or a preset algorihm like "Triangle".
    checkAndThreshold(thresh);
    
    // Filter out small particles and make mask. Set min size of epithlium. Default is 150 pixels. 
    run("Analyze Particles...", "size=" + epiSize + "-Infinity show=Masks");
    
    //erode x2 then dilate x2 to get rid of small attachments to larger epi. You can set number of erosions/dilations as desired. 
    for(j=0;j<erodeNum;j++) run("Erode");
    for(j=0;j<dilateNum;j++) run("Dilate");
	
	run("Analyze Particles...", "size=" + epiSize + "-Infinity show=Masks");
	
    //Built in delay so user can check if happy with result. Also prints number of current image for user to note down if any issues with image. 
    wait(delayTime);
	Array.show(intArray, a);
	print(i + 1);
	selectWindow("Log");
	wait(delayTime + 2000);
	
    // Get image name without extension
    image_name = File.nameWithoutExtension;

    // Create output directory if it doesn't exist
    dir_image_output = dirOutput + "/" + image_name;

    // Save the mask
    saveAs("Tiff", dir_image_output + "_EP.tiff");

    // Close the image
    run("Close All");
}
Array.show(intArray, a);

function checkAndThreshold(thresh) {
    if (isNaN(thresh + 0)) {
        // The variable 'thresh' is a string.
        setAutoThreshold(thresh + " dark no-reset");
    } else {
        // The variable 'thresh' is a number.
        setThreshold(thresh, 65535, "raw");
    }
}

