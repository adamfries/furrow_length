
Author: Adam Fries

Institution: University of Oregon

email: afries2@uoregon.edu

Date: 2023/05/01

This code is available for free use and without guarantees. 

furrow_detection.py

## GENRAL DESCRIPTION:
This code reads in a directory of single channel, single plane (grayscale) movies of C. elegans oocyte cortex
and identifies points of ingression and their lengths for all time points. To isolate the chromosome associated 
furrow, the code uses a single channel, single plane (MIP is useful) movies of the oocyte chromosomes. The results 
are stored into csv files and images that highlight the ingression points.

## PYTHON LIBRARIES REQUIRED:
`pandas`
`scikit-image`
`scipy`
`numpy`
`matplotlib`
`tifffile`

## INPUT DATA ORGANIZATION REQUIREMENTS:

1. create a folder for all conditions
2. within this folder, create subfolders for each condition and a folder named `RAW`
3. place the single channel tif movies in the appropriate condition folders
4. within the `RAW` folder create subfolders with identical names to each condition folder
5. within the RAW/condition subfolders, place the chromosome tif movies in the appropriate condition folders

## RUNNING THE CODE:
``python3 furrow_detection.py all_conditions/condition1``

where ``all_conditions`` is your folder containing all conditions subfolders, 
and ``condition1`` is the subfolder for condition1 containing your cortex movies

## KNOWN ISSUES:
The code is sensitive to other objects in the field, so cropping out everything outside the cortex
is necessary for the code to work properly.
