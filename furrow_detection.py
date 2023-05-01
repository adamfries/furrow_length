import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.filters import (median, 
							gaussian, 
							threshold_yen)
from skimage.morphology import (disk, 
								remove_small_objects, 
								remove_small_holes,
								convex_hull, 
								binary_closing, 
								square, 
								dilation, 
								erosion, 
								closing)
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from skimage.measure import (find_contours, 
							label, 
							regionprops_table, 
							points_in_poly)
from skimage import util
import sys
import os
import tifffile as tf
from scipy.spatial import Delaunay
import scipy.spatial as spatial
import itertools
import copy
import re
 
#plt.rcParams['text.usetex'] = True

def convex_hull_image(image):
    """Compute the convex hull image of a binary image.
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.
    Parameters
    ----------
    image : (M, N) array
        Binary input image. This array is cast to bool before processing.
    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.
    References
    ----------
    .. [1] http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/
    """
    if image.ndim > 2:
        raise ValueError("Input must be a 2D image")

    if Delaunay is None:
        raise ImportError("Could not import scipy.spatial.Delaunay, "
                          "only available in scipy >= 0.9.")

    # Here we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual hull.
    coords = convex_hull.possible_hull(image.astype(np.uint8))
    N = len(coords)

    # Add a vertex for the middle of each pixel edge
    coords_corners = np.empty((N * 4, 2))
    for i, (x_offset, y_offset) in enumerate(zip((0, 0, -0.5, 0.5),
                                                 (-0.5, 0.5, 0, 0))):
        coords_corners[i * N:(i + 1) * N] = coords + [x_offset, y_offset]

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.Delaunay, so we remove them.
    coords = convex_hull.unique_rows(coords_corners)

    # Subtract offset
    offset = coords.mean(axis=0)
    coords -= offset

    # Find the convex hull
    chull = Delaunay(coords).convex_hull
    v = coords[np.unique(chull)]

    # Sort vertices clock-wise
    v_centred = v - v.mean(axis=0)
    angles = np.arctan2(v_centred[:, 0], v_centred[:, 1])
    v = v[np.argsort(angles)]

    # Add back offset
    v += offset

    # For each pixel coordinate, check whether that pixel
    # lies inside the convex hull
    mask = convex_hull.grid_points_in_poly(image.shape[:2], v)

    return mask, v


def process_img(img):
	''' 
		apply median filer to the image to help with segmentation, 
	 	set as uint16 to match ImageJ output. Assumes an inverted image
		as the input. If not, comment the line below and uncomment the line
		below that
	'''
	#img = util.invert(median(img.astype('uint16'), square(2)))
	img = median(img.astype('uint16'), square(2))
	
	## threshold the image using yen and return the boolean result
	thresh = threshold_yen(img)
	return img, thresh

def get_contour_hull(binary, yx, xs):
	## positive orientation = high to match orientation of convex hull
	cont = find_contours(binary, 0.5, positive_orientation = 'high')

	'''
		find_contours will find all objects in the image, we only
		want the largest and need to ignore anything small for the 
		contours:
		Shoelace formula for finding the area of a polygon in 
		cartesian coordinates and filter out everything but the 
		largest - assume the largest object is the object of interest
	'''
	max_c_area = 0
	max_c = 0

	for c in cont:
		cx = c[:, 0]
		cy = c[:, 1]
		c_area = 0.5*np.abs(np.dot(cx, np.roll(cy, 1)) - 
					np.dot(cy, np.roll(cx, 1)))
		if c_area > max_c_area:
			max_c_area = c_area
			max_c = c
	if not isinstance(max_c, int ):
		cx = max_c[:, 0]
		cy = max_c[:, 1]

	
	chull, verts = convex_hull_image(binary)

	## return the convex hull, the coordinates of the vertices, and the 
	##	x,y-coodinates of the contour
	return chull, verts, cx, cy

def find_common_coords(list1, list2):
	''' 
		used to map the vertices of the convex hull to the coordinates
		of the contour - this is a clunky way to do it - uses two 
		k-dimensional trees to find closest pairs. The convex hull 
		uses the contour to establish vertices, so perhaps better to use
		that in the future
	'''
	kd_tree1 = spatial.cKDTree(list1)
	kd_tree2 = spatial.cKDTree(list2)
	ind = kd_tree1.query_ball_tree(kd_tree2, r = 0.25)
	ind = sorted(list(itertools.chain(*filter(None, ind))))
	
	return list2[ind], ind


def find_dist_from_line(p1, p2, p3):
	''' 
		find max distance between line defined by hull vertices 
		and contour: d = ||PQ x u|| / ||u||
		PQ = p2 - p1, u = p3 - p1
	    p1 is vertex 1, p2 is vertex 2, p3 is the contour point
	 	p2, p2, p3 all need to be numpy arrays
	'''
	if np.linalg.norm(p2 - p1) != 0:
		return np.abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
	else:
		return 0		

def chromo_segmentation(raw_img):
	'''
		segment the MAX proj movie of the chromosomes using Yen threshold
		the closing and remove_small_objects is used to clean up and remove
		any small unwanted holes or objects
	'''
	ts, ys, xs = raw_img.shape
	binary_chromo = np.zeros((ts, ys, xs))
	for t in range(ts):
		max_project = raw_img[t, :, :]
		max_yen = threshold_yen(max_project)
		bw = closing(max_project > max_yen, disk(3))
		binary_chromo[t, :, :] = remove_small_objects(clear_border(bw), 10)
	
	return binary_chromo

def main():
	''' 
		begin main: indir is the input directory of the moveis to be 
		measured. The following code also assumes that there exists a
		directory with the input folder named 'RAW' which contains
	 	identically named subfolders as the subfolders contained in indir.
	'''
	indir = sys.argv[1]

	maindir = indir.split('/')[:-1][0]
	
	files = os.listdir(indir)
	
	## define constants

	## pix_scale is the image pixel scale for the 100x objecitve used 
	##	on the Andor Spinning disk with the Andor EMCCD 16 um pixels and 
	## 	is 0.147 um/pix, I have changed this to 1 for general use 
	pix_scale = 1.0
	pix_scale = 0.147

	## tol is the tolerance argument used for np.isclose() in mapping the 
	##	areas of the ingression to the associated furrow coordinates in 
	##	pixels and must be within 0.6 pixels.  
	tol = 0.6

	## the following are the input for scipy's find_peaks
	min_dist = 3.0
	min_sep = 15.0
	min_prom = 3.0

	## image frame size which encloses the contour overlays
	bigx, bigy = 600, 600

	## begin looping though files within each condition folder
	for infile in files:
		overlay_img = np.zeros((bigy, bigx))
		if infile.endswith('.tif') and not infile.startswith('.'):
			## create a directory to save plots based on filename
			savepath = (os.path.splitext(infile))[0]
			savepath = indir + '/' + savepath
			
			print('Reading file:', infile)
			if not os.path.exists(savepath):
				os.mkdir(savepath)
			else:
				continue

			## create an overlays folder to save the grayscale stills
			##	with magental dot overlays of furrow peaks
			saveoverlaypath = os.path.join(savepath, 'overlays')
			if not os.path.exists(saveoverlaypath):
				os.mkdir(saveoverlaypath)

			savename = os.path.basename(os.path.normpath(savepath))	
	
			## read in the tif
			imgstack = tf.imread(indir + '/' + infile).astype('float64')
			ts, ys, xs = imgstack.shape	

			## read in the raw tif directories for chromosome segmentation
			cdir = indir.split('/')[-1]		
			raw_files = os.listdir(os.path.join(maindir, 'RAW', cdir))	

			## use regex to pattern match the hand-segmented condition
			##	folder with the RAW max projection chromosome folder
			patn1 = r'(_slice\d+)'
			img_pre1 = re.sub(
							  patn1, '', '_'.join(
							  os.path.splitext(
							  infile)[0].split('_')[:-1]))
			for r in raw_files:
				#if r.endswith('RAW.tif'):	
				raw_file = '_'.join(os.path.splitext(r)[0].split('_')[:-2])
				## read in the RAW chromosome file if a match is found
				#print(raw_file, img_pre1)
				if (raw_file == img_pre1): 
					raw_bin = chromo_segmentation(
												  tf.imread(
												  os.path.join(
												  maindir, 'RAW', cdir, r)))
				elif (raw_file != img_pre1):
					continue
				else:
					## if the RAW match isn't found exit the software
					print('Error: Unable to find RAW match for ', raw_file)
					raw_bin = ''
					exit()

			## create empty img for peak overlay
			peaks_overlay = np.zeros((ys, xs))

			## small objects are anything smaller than 5000 pixels
			##	or have a radius smaller than 40 pixels
			small_object = 5000

			## create an empty list to populate all distances, areas, times
			all_dists = []
			all_indices = []
			all_areas = []
			time_pts = []
			
			## for each image file (movie) loop through each frame
			for i in range(ts):
				print('Analyzing time point:', i)
				img = imgstack[i, :, :]
				## threshold the image 
				img1, thresh = process_img(img)
				img2 = copy.deepcopy(img1)
				## perform morphological closing 
				binary = closing(img1 > thresh, square(4))
				## remove all objects that are not the oocyte
				binary = remove_small_objects(clear_border(binary), 
											  small_object)
				binary = remove_small_holes(clear_border(binary), 
											small_object)
				''' 
					perform binary dilation and invert, remove the 
					non-zero object touching the border. This step 
					enables us to create the contour based on the interior
					of the ooctyte (and not just the exterior). This way
					the long narrow furrows are fully captured and can 
					then be measured
				'''
				binary = dilation(binary, disk(2))
				binary = util.invert(binary)
				labels, num = label(binary, 
									connectivity = 2, 
									return_num = True)	
				binary = clear_border(labels)
				#fig, ax = plt.subplots()
				#ax.imshow(binary)
				#plt.show()
				
				'''
					double check that the segmented object is larger than 
					our small_object cutoff. If so, calculate the contour
					model and convex hull. If not, keep track of the
					failed time point and move on to the next. We expect
					not every timepoint to segment properly, but if more
					than a couple time poionts fail, it can help to 
					manually segment the oocytes more carefully.	
				'''
				if np.sum(binary) > small_object:	
					chull, verts, cx, cy = get_contour_hull(binary, ys, xs)
				else:
					print('Warning: segmentation failed.'  
						  'Skipping time point:', i)
					with open(indir + '/problem_images.txt', 'a') as f:
						f.write(infile + 
								'\t' + 
								'error failed segmentation' + 
								'\tt' + 
								str(i) + 
								'\n')
					continue
				
				## get the areas of the furrows by finding the 
				##	difference between the convex hull and contour	
				cont_hull = binary_fill_holes(binary)
				diff = cont_hull < chull
				area_img = np.zeros((ys, xs))
				diff = remove_small_objects(diff, 1)	
				labels = label(diff)
				props = regionprops_table(labels, 
										  properties = ['area', 'coords'])	
				areas = props['area']
				coords = props['coords']	

				## to make a closed loop we need the first coords at the 
				##	beginning and the end
				verts = np.append(verts, [verts[0]], axis = 0)
				cx = np.append(cx, cx[0])
				cy = np.append(cy, cy[0])
				contour_xy = np.stack((cx, cy), axis = 1)

				## add up the contours to form a time-lapse overlay
				offsety = int(bigy/2 - xs/2)
				offsetx = int(bigx/2 - ys/2)
				overlay_img[cx.astype(int) + offsetx, 
							cy.astype(int) + offsety] = 255

				## find the common set of coordinates between the 
				##  contour and convex hull vertices as a way to loop 
				##  around the contours
				c_verts, ind = find_common_coords(verts, contour_xy)
				
				## if no common coodinates are found between the convex
				##	hull and the contour, something went wrong with the 
				##	segmentation, keep track and skip the the time point	
				if len(c_verts) < 1:
					print('Warning: contour does not match convex hull!' 
						  'Skipping time point:', i)
					with open(indir + '/problem_images.txt', 'a') as f:
						f.write(infile + 
								'\t' + 
								'error contour mismatch' + 
								'\tt' + 
								str(i) + 
								'\n')
					continue
				## accounting
				c_verts = np.transpose(c_verts)

				## create an empty list to populate with the distances 
				##	found for each time point
				max_dists = []
				max_contx = []
				max_conty = []
				max_areas = []
				max_indices = []
		
				## loop around the hull vertices
				for j in range(len(c_verts[0, :]) - 1):
					## grab the sections of contour points between 
					##	hull vertices
					current_contx = cx[ind[j]:ind[j + 1]]
					current_conty = cy[ind[j]:ind[j + 1]]

					## define the orthogonal line that makes the 
					##	section of the hull
					x0, x1 = c_verts[0, j], c_verts[0, j + 1]
					y0, y1 = c_verts[1, j], c_verts[1, j + 1]
					p1 = np.array([y0, x0])
					p2 = np.array([y1, x1])
					max_d = []
					max_i = [np.nan]	
					
					## find all the distance values between vertices
					p3 = list(zip(current_conty, current_contx))
					ds = find_dist_from_line(p1, p2, p3)
					
					## ensur that there are at least 3 points which
					## make up the two vertices and 1 furrow. We use 
					## scipy's find_peaks here for situations where there
					## are multiple furrows between hull vertices 
					if len(p3) > 2:
						peaks, peak_props = find_peaks(ds, 
											height = min_dist, 
											distance = min_sep, 
											prominence = min_prom)
				
						## make sure the peaks exist before continuing	
						if peak_props['peak_heights'].size > 0:
							peaks_y = current_conty[peaks]
							peaks_x = current_contx[peaks]
							max_dists.extend(peak_props['peak_heights']*pix_scale)
							max_conty.extend(peaks_y)
							max_contx.extend(peaks_x)
							
							## track the peak positions along the contour 
							for c, pairs in enumerate(coords):
								for yy, xx in pairs:
									if np.isclose(peaks_y[0], xx, atol = tol) and np.isclose(peaks_x[0], yy, atol = tol):
										max_areas.append(areas[c]*pix_scale*pix_scale)		
										max_indices.append(c)
					
				## check that the furrows are not within the chromosome
				## projection, if so, save furrow info to textfile	
				chrome_tolerance = 22
				
				#print('time: ', i)
				if len(raw_bin) > 0:
					chrome_coords = np.argwhere(raw_bin[i, :, :] > 0)
				else:
					chrome_coords = ''

				if len(chrome_coords) > 0:
					with open(os.path.join(savepath, savename + '_main_furrows.txt'), 'a') as fmf:
						for z, (x, y) in enumerate(zip(max_contx, max_conty)):
							chrome_dist = np.min(np.sqrt((y - chrome_coords[:, 1])**2 + 
								(x - chrome_coords[:, 0])**2))
							if chrome_dist < chrome_tolerance:
								main_furrow_dist = max_dists[z]
								## save main furrow text file: time, x, y, dist
								fmf.write(str(i) + '\t' + str(x) + '\t' + str(y) + '\t' + str(main_furrow_dist) + '\n')
				
				if not max_dists:
					all_dists.append([0])
					all_areas.append([0])
					time_pts.append(i + 1)
				else:
					all_dists.append(max_dists)
					all_areas.append(max_areas)	
					all_indices.append(max_indices)
					time_pts.append(i + 1)

				for a in coords[max_indices]:
					area_img[a[:, 0], a[:, 1]] = 255

				fig, ax = plt.subplots()
				ax.plot(max_conty, max_contx, 
					'o', color = 'magenta')
				ax.imshow(img2, cmap = 'gray')
				plt.axis('off')
				if i < 10:
					plt.savefig(os.path.join(saveoverlaypath, savename + '_overlay_timepoint_0' + str(i) + '.png'))
				else:
					plt.savefig(os.path.join(saveoverlaypath, savename + '_overlay_timepoint_' + str(i) + '.png'))
				plt.close()

				## plot the convex hull and binary image
				fig, ax = plt.subplots(3, 1, figsize = (5, 8))
				ax[2].imshow(area_img, cmap = plt.gray())
				ax[1].plot(max_conty, max_contx, '*', markersize = 10, color = 'magenta')
				ax[1].scatter(c_verts[1, :], c_verts[0, :], color = 'black', alpha = 0.7)
				ax[1].plot(cy, cx)
				ax[1].invert_yaxis()
				ax[0].imshow(binary, cmap = plt.gray())
				ax[1].set_xlim(0, xs)
				ax[1].set_ylim(ys, 0)
				ax[0].set_aspect('equal')
				ax[1].set_aspect('equal')
				ax[2].set_aspect('equal')

				plt.savefig(os.path.join(savepath, savename + 'time_point' + str(i) + '.png'))
				plt.close()

			## suppress warnings for empty slices / divide by zeros, etc. 
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', category=RuntimeWarning)
				num_dists = [len(i) for i in all_dists]
				mean_dists = np.array([np.mean(i) for i in all_dists])
				std_dists = np.array([np.std(i) for i in all_dists])
				mean_areas = np.array([np.mean(i) for i in all_areas])
				std_areas = np.array([np.std(i) for i in all_areas])

			dists_df = pd.DataFrame(all_dists)
			dists_df = dists_df.T
			dists_df.columns = time_pts
			dists_df.to_csv(os.path.join(savepath, savename + '_lengths_per_time_point.csv'))

			areas_df = pd.DataFrame(all_areas)
			areas_df = areas_df.T
			areas_df.columns = time_pts
			areas_df.to_csv(os.path.join(savepath, savename + '_areas_per_time_point.csv'))

			rcParams['font.family'] = 'Times New Roman'
			rcParams['font.size'] = 16

			fig, ax = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
			ax[0].plot(time_pts, mean_dists, 'ko', label = 'Mean')
			ax[0].fill_between(time_pts, mean_dists - std_dists, mean_dists + std_dists, color = 'black', alpha = 0.35, label = 'Std.')
			ax[1].plot(time_pts, num_dists, 'ko')
			ax[2].plot(time_pts, mean_areas, 'ko', label = 'Mean')
			ax[2].fill_between(time_pts, mean_areas - std_areas, mean_areas + std_areas, color = 'black', alpha = 0.35, label = 'Std.')
			ax[0].set_ylabel('Furrow Length ($\mu$m)')
			ax[1].set_ylabel('No. Furrows')
			ax[2].set_ylabel('Area ($\mu$m$^2$)')
			ax[2].set_xlabel('Time Index')

			## get the limits
			_, maxy0 = ax[0].get_ylim()
			_, maxy1 = ax[1].get_ylim()
			_, maxy2 = ax[2].get_ylim()
			plt.close()

			plt.subplots_adjust(hspace = .0)
			ax[0].yaxis.get_major_locator().set_params(integer = True)
			ax[1].yaxis.get_major_locator().set_params(integer = True)
			ax[2].yaxis.get_major_locator().set_params(integer = True)
		
			plt.savefig(os.path.join(savepath, savename + '_stats.png'))

			figo, axo = plt.subplots()
			axo.imshow(overlay_img, cmap = 'gray')
			plt.axis('off')
			plt.savefig(os.path.join(savepath, savename + '_temporal_overlays.png'))
			plt.close()
							
main()
