import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.filters import (median, gaussian, 
							threshold_otsu, 
							threshold_yen,
							threshold_li)
from skimage.morphology import (disk, 
								skeletonize, 
								remove_small_objects, 
								remove_small_holes,
								convex_hull, 
								binary_closing, 
								thin, 
								medial_axis,
								square, 
								dilation)
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from skimage.measure import find_contours, label, regionprops_table
from skimage import util
from skimage.segmentation import clear_border
import sys
import os
import tifffile as tf
from scipy.spatial import Delaunay, distance
import scipy.spatial as spatial
import itertools
from functools import partial
import copy
import cProfile
import pstats

 
plt.rcParams['text.usetex'] = True


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
	## apply median filer to image to help with segmentation, set as uint16 to match 
	## ImageJ output

	img = util.invert(median(img.astype('uint16'), square(2)))
	## threshold the image using yen and return the boolean result
	thresh = threshold_yen(img)
	thresh_li = threshold_li(img)
	#return img > thresh
	return img, thresh, thresh_li

def get_contour_hull(binary, yx, xs):
	## positive orientation = high to match orientation of convex hull
	cont = find_contours(binary, 0.5, positive_orientation = 'high')

	'''
		find_contours will find all objects in the image, we only
		want the largest and need to ignore anything small for the 
		contours:
		Shoelace forula for finding the area of a polygon in 
		cartesian coordinates and filter out everything but the 
		largest - assume the largest object is the object of interest
	'''
	max_c_area = 0
	max_c = 0

	for c in cont:
		cx = c[:, 0]
		cy = c[:, 1]
		c_area = 0.5*np.abs(np.dot(cx, np.roll(cy, 1)) - np.dot(cy, np.roll(cx, 1)))
		if c_area > max_c_area:
			max_c_area = c_area
			max_c = c
	if not isinstance(max_c, int ):
		cx = max_c[:, 0]
		cy = max_c[:, 1]

	
	chull, verts = convex_hull_image(binary)

	return chull, verts, cx, cy

def find_common_coords(list1, list2):
	xxx = 2
	if xxx == 1:
		c_verts = []
		ind = []
		for i, l1 in enumerate(list2):
			for x2, y2 in list1:
			##if l1[0] == x2 and l1[1] == y2:
				if np.isclose(l1[0], x2, atol = 0.25) and np.isclose(l1[1], y2, atol = 0.25):
					c_verts.append((l1[0], l1[1]))
					ind.append(i)
		return c_verts, ind
	else:
	## fast way?
		#x1 = list1[:, 0]
		#y1 = list1[:, 1]
		#x2 = list2[:, 0]
		#y2 = list2[:, 1]
	
		##ind = np.array([True for i in x2 if np.isclose(i, x1, 0.25).any()]) & np.array([True for i in y2 if np.isclose(i, y1, 0.25).any()])
		#ind = np.logical_and([True for i in x2 if np.isclose(i, x1, 0.25).any()], np.array([True for i in y2 if np.isclose(i, y1, 0.25).any()]))
		kd_tree1 = spatial.cKDTree(list1)
		kd_tree2 = spatial.cKDTree(list2)
		ind = kd_tree1.query_ball_tree(kd_tree2, r = 0.25)
		#ind = [kd_tree1.query_ball_point([x, y], 0.25) for x, y in list2]
		
		ind = sorted(list(itertools.chain(*filter(None, ind))))
		
		return list2[ind], ind


def find_dist_from_line(p1, p2, p3):
	## find max distance between line defined by hull vertices 
	##	and contour: d = ||PQ x u|| / ||u||
	##  PQ = p2 - p1, u = p3
	## p1 is vertex 1, p2 is vertex 2, p3 is the contour point
	## p2, p2, p3 all need to be numpy arrays
	if np.linalg.norm(p2 - p1) != 0:
		return np.abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
	else:
		return 0		

def main():
	indir = sys.argv[1]

	files = os.listdir(indir)
	#excel_df = pd.read_excel(sys.argv[2])

	plot_flag = 1#sys.argv[3]

	pix_scale = 0.147
	tol = 0.6
	min_dist = 3.0
	min_sep = 15.0
	min_prom = 3.0
	
	bigx, bigy = 600, 600



	for infile in files:
		## get the time dependent annotations
		#ana_b = excel_df[excel_df.Name == infile]['Anaphase B']
		#mei2 = excel_df[excel_df.Name == infile]['Meiosis II']
		#gcfi = excel_df[excel_df.Name == infile]['GCFi']
		#gcff = excel_df[excel_df.Name == infile]['GCFf']

		overlay_img = np.zeros((bigy, bigx))
		if infile.endswith('.tif') and not infile.startswith('.'):
			## create a directory to save plots based on filename
			savepath = (os.path.splitext(infile))[0]
			savepath = indir + '/' + savepath
			
			if not os.path.exists(savepath):
				os.mkdir(savepath)
			else:
				continue

			saveoverlaypath = os.path.join(savepath, 'overlays')
			if not os.path.exists(saveoverlaypath):
				os.mkdir(saveoverlaypath)

			savename = os.path.basename(os.path.normpath(savepath))			
				
			## read in the tif
			print(indir, infile)
			imgstack = tf.imread(indir + '/' + infile).astype('float64')
			ts, ys, xs = imgstack.shape
			
			## create empty img for peak overlay
			peaks_overlay = np.zeros((ys, xs))

			## small objects are anything smaller than 5000 pixels
			small_object = 5000

			## create an empty list to populate all distances for all times
			all_dists = []
			all_indices = []
			all_areas = []
			time_pts = []


			## TODO (check that we still need to do this) ignoring the last time point for now
			for i in range(ts):
				print('Time point:', i)

				#img = util.invert(imgstack[i, :, :])
				img = imgstack[i, :, :]
				img1, thresh, thresh_li = process_img(img)
				img2 = copy.deepcopy(img1)
				binary = img1 > thresh
				#binary = img1 > thresh_li
				#binary = img1 > 500
				binary_static = img1 > 52000
				
				#fig, ax = plt.subplots()
				#ax.imshow(binary_static)
				#plt.show()


				
				## try removing small objects here from binary before calculating the 
				##	convex hull to prevent small objects from acting as vertices
				binary = remove_small_objects(binary, small_object, in_place = True)
				binary = remove_small_holes(binary, small_object, in_place = True)
				binary = dilation(binary, disk(2))
				
				binary = util.invert(binary)
				labels, num = label(binary, connectivity = 2, return_num = True)
				
				
				if num < 2:
					print('Warning: unable to find contiguous edge: trying static threshold:', i)
					with open(indir + '/problem_images.txt', 'a') as f:
						f.write(infile + '\t' + 'error not contiguous' + '\tt' + str(i) + '\n')
					#continue
					binary = remove_small_objects(binary_static, small_object, in_place = True)
					
					binary = util.invert(binary)
					labels, num = label(binary, connectivity = 2, return_num = True)
					if num < 2:
						print('Static threshold failed: skipping time point')
						continue



				binary =clear_border(labels)

				if np.sum(binary) > 100:	
					chull, verts, cx, cy = get_contour_hull(binary, ys, xs)
				else:
					print('Warning: segmentation failed. Skipping time point:', i)
					with open(indir + '/problem_images.txt', 'a') as f:
						f.write(infile + '\t' + 'error failed segmentation' + '\tt' + str(i) + '\n')
					continue
				
				## plot the chull rq to remind me what it is
				## we want the difference between the chull and shape defined by 
				##	cx, cy. It turns out that the convex hull uses the small objects
				##	we need to filter those out so that the convex hull only uses the largest
				##	object
				cont_hull = binary_fill_holes(binary)
				diff = cont_hull < chull
				area_img = np.zeros((ys, xs))

				##	assume an equilateral triangle
				diff = remove_small_objects(diff, 1)	
				labels = label(diff)
				props = regionprops_table(labels, properties = ['area', 'coords'])	
				areas = props['area']
				coords = props['coords']	

				## make a closed loop we need the first coords at the 
				##	beginning and the end
				verts = np.append(verts, [verts[0]], axis = 0)
				cx = np.append(cx, cx[0])
				cy = np.append(cy, cy[0])
				contour_xy = np.stack((cx, cy), axis = 1)

				## add up the contours to form an time-lapse overlay
				offsety = int(bigy/2 - xs/2)
				offsetx = int(bigx/2 - ys/2)
				overlay_img[cx.astype(int) + offsetx, cy.astype(int) + offsety] = 255

				## find the common set of coordinates between the contour and 
				##	convex hull vertices as a way to loop around the contours
				
				''' time test 1 - common coords bottle neck
				'''
				#profile = cProfile.Profile()
				#profile.enable()


				##	to find the defects
				c_verts, ind = find_common_coords(verts, contour_xy)
				'''end test 1 file
					write verts and contour_xy to file for testing
				'''
				#np.savetxt('verts.txt', verts)		
				#np.savetxt('contour.txt', contour_xy)			

				#profile.disable()
				#ps = pstats.Stats(profile)
				#ps.print_stats()
				#jhhjhjh
				
				## TODO transform c_verts but check that we get the same c_verts between methods!!
					
				if len(c_verts) < 1:
				#if not c_verts:
					print('Warning: contour does not match convex hull! Skipping time point:', i)
					with open(indir + '/problem_images.txt', 'a') as f:
						f.write(infile + '\t' + 'error contour mismatch' + '\tt' + str(i) + '\n')
					continue
				'''
				c_verts = np.array(list(map(list, zip(*c_verts))))
				'''
				c_verts = np.transpose(c_verts)
				

				## create an empty list to populate with the distances found for each time point
				max_dists = []
				max_contx = []
				max_conty = []
				max_areas = []
				max_indices = []
				## loop around the hull vertices
				for j in range(len(c_verts[0, :]) - 1):
					## grab the sections of contour points between hull vertices
					current_contx = cx[ind[j]:ind[j + 1]]
					current_conty = cy[ind[j]:ind[j + 1]]
					'''
					'''
					## define the orthogonal line that makes the section of the hull
					x0, x1 = c_verts[0, j], c_verts[0, j + 1]
					y0, y1 = c_verts[1, j], c_verts[1, j + 1]
					
					p1 = np.array([y0, x0])
					p2 = np.array([y1, x1])
					
					max_d = []
					max_i = [np.nan]	
					
					## find all the distance values between vertices
					## make p3 an array of pairs
					p3 = list(zip(current_conty, current_contx))
					ds = find_dist_from_line(p1, p2, p3)
				
					if len(p3) > 2:
						peaks, peak_props = find_peaks(ds, 
											height = min_dist, 
											distance = min_sep, 
											prominence = min_prom)
					
						if peak_props['peak_heights'].size > 0:
							peaks_y = current_conty[peaks]
							peaks_x = current_contx[peaks]
							max_dists.extend(peak_props['peak_heights']*pix_scale)
							max_conty.extend(peaks_y)
							max_contx.extend(peaks_x)

							for c, pairs in enumerate(coords):
								for yy, xx in pairs:
									if np.isclose(peaks_y[0], xx, atol = tol) and np.isclose(peaks_x[0], yy, atol = tol):
										max_areas.append(areas[c]*pix_scale*pix_scale)		
										max_indices.append(c)
							
				all_dists.append(max_dists)
				all_areas.append(max_areas)	
				all_indices.append(max_indices)
				time_pts.append(i + 1)
								
				if not max_dists:
					all_dists.append([0])
					all_areas.append([0])
					time_pts.append(i + 1)

				for a in coords[max_indices]:
					area_img[a[:, 0], a[:, 1]] = 255

				plot_glag = 0	
				if plot_flag:				
					## plot the peaks_overlay over the raw image
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
				#time_pts = range(1, len(num_dists) + 1)

			dists_df = pd.DataFrame(all_dists)
			dists_df = dists_df.T
			dists_df.columns = time_pts
			dists_df.to_csv(os.path.join(savepath, savename + '_lengths_per_time_point.csv'))

			areas_df = pd.DataFrame(all_areas)
			areas_df = areas_df.T
			areas_df.to_csv(os.path.join(savepath, savename + '_areas_per_time_point.csv'))

			rcParams['font.family'] = 'Times New Roman'
			rcParams['font.size'] = 16

			
			## plot the mean defect lengths, areas, number versus time
			##ax.annotate('local max', xy=(3, 1),  xycoords='data',
            ##	xytext=(0.8, 0.95), textcoords='axes fraction',
            ##	arrowprops=dict(facecolor='black', shrink=0.05),
            ##	horizontalalignment='right', verticalalignment='top',
            ##	)

	
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
			

			## annotate anaphase and meiosis
			#ax[0].plot([ana_b, ana_b], [0, maxy0], 'k--')
			#ax[1].plot([ana_b, ana_b], [0, maxy1], 'k--')
			#ax[2].plot([ana_b, ana_b], [0, maxy2], 'k--')
			
			#ax[0].plot([mei2, mei2], [0, maxy0], 'k:')
			#ax[1].plot([mei2, mei2], [0, maxy1], 'k:')
			#ax[2].plot([mei2, mei2], [0, maxy2], 'k:')
			

			plt.subplots_adjust(hspace = .0)
			#ax[0].grid()
			#ax[1].grid()
			#ax[2].grid()
			#ax[0].legend(loc = 'upper left')
			ax[0].yaxis.get_major_locator().set_params(integer = True)
			ax[1].yaxis.get_major_locator().set_params(integer = True)
			ax[2].yaxis.get_major_locator().set_params(integer = True)
		
						#ax[0].annotate('Anaphase B', xy=(ana_b, 3),  xycoords='data',
            #	xytext=(0.2, 1.1), textcoords='axes fraction',
            #	arrowprops=dict(facecolor='black', shrink=0.01,width = 1),
            #	horizontalalignment='right', verticalalignment='top')
			#ax[0].annotate('Meiosis II', xy=(mei2, 3),  xycoords='data',
            #	xytext=(0.4, 1.1), textcoords='axes fraction',
            #	arrowprops=dict(facecolor='black', shrink=0.01, width = 1),
            #	horizontalalignment='right', verticalalignment='top')			
			#ax[0].annotate('GCFi', xy=(gcfi, 3),  xycoords='data',
            #	xytext=(0.6, 1.1), textcoords='axes fraction',
            #	arrowprops=dict(facecolor='black', shrink=0.01, 
			#	width = 1),
           	#	horizontalalignment='right', verticalalignment='top',
           	#	)
			#ax[0].annotate('GCFf', xy=(gcff, 3),  xycoords='data',
            #	xytext=(0.8, 1.1), textcoords='axes fraction',
            #	arrowprops=dict(facecolor='black', shrink=0.01, 
			#	width = 1),
            #	horizontalalignment='right', verticalalignment='top',
            #	)


			plt.savefig(os.path.join(savepath, savename + '_stats.png'))

			figo, axo = plt.subplots()
			axo.imshow(overlay_img, cmap = 'gray')
			plt.axis('off')
			plt.savefig(os.path.join(savepath, savename + '_temporal_overlays.png'))
			#np.savetxt('cverts_slow.txt', c_verts)		
		
							
main()
