import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage.filters import median, gaussian, threshold_otsu, threshold_li, threshold_local, threshold_triangle
from skimage.morphology import disk, skeletonize, remove_small_objects, convex_hull, binary_closing, thin, medial_axis
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks
from skimage.measure import find_contours, label, regionprops_table
from skimage import util
import sys
import os
import tifffile as tf
from scipy.spatial import Delaunay, distance
import itertools
from functools import partial

 
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


def process_img(img, sigma = 40):
	## process_img subtracts background and remove negative values

	## apply median filer to image to help with segmentation
	img = median(img, disk(4))

	## the background is characterized using a low-pass filter
	backgrd_img = gaussian(img, sigma = sigma)
	
	## subtract the background
	img -= backgrd_img

	## assign any negative values to zero
	img[img < 0] = 0 

	## threshold the image using otsu and return the boolean result
	thresh = threshold_li(img)
	#thresh = threshold_local(img, 15)	

	return img > thresh


def get_contour_hull(binary, yx, xs):
	#binary = binary_closing(skeletonize(binary, method = 'lee'))
	#skel = binary_closing(thin(binary))
	#skel = binary_closing(medial_axis(binary))

	#fig, ax = plt.subplots()
	#ax.imshow(skel)
	#plt.show()
	#skel = binary_fill_holes(skel)
	## remove any objects smaller than 100 pixels
	#skel = skel.astype('uint8')

	#binary = skel
	#points = np.argwhere(skel > 0)
	## positive orientation = high to match orientation of convex hull
	cont = find_contours(binary, 0.5, positive_orientation = 'high')

	## find_contours will find all objects in the image, we only
	## 	want the largest and need to ignore anything small for the contours:
	## 	Shoelace forula for finding the area of a polygon in cartesian coordinates
	## 	and filter out everything but the largest - assume the largest object is the 
	##	object of interest
	max_c_area = 0

	for c in cont:
		cx = c[:, 0]
		cy = c[:, 1]
		c_area = 0.5*np.abs(np.dot(cx, np.roll(cy, 1)) - np.dot(cy, np.roll(cx, 1)))
		if c_area > max_c_area:
			max_c_area = c_area
			max_c = c

	#c = cont[0]
	cx = max_c[:, 0]
	cy = max_c[:, 1]

	
	## create an image for the contour output at twice the resolution
	##	the out put of find contours has a resolution of 0.5 pixels
	#c_image = np.zeros((ys, xs)).astype('int')
	#cx_int = [int(2*x) for x in cx]
	#cy_int = [int(2*y) for y in cy]
	#c_image[cx_int, cy_int] = 255


	chull, verts = convex_hull_image(binary)
	#chull = chull.astype('uint8')

	return chull, verts, cx, cy

def find_common_coords(list1, list2):
	c_verts = []
	ind = []
	for i, l1 in enumerate(list2):
		for x2, y2 in list1:
			if l1[0] == x2 and l1[1] == y2:
				c_verts.append((l1[0], l1[1]))
				ind.append(i)
	return c_verts, ind


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

def find_defects(d_curr, d_past):
	if d_curr < d_past:
		return d_past

def find_min_defects(d_curr, d_past):
	if (d_curr > d_past):
		return d_past


def main():
	indir = sys.argv[1]

	files = os.listdir(indir)

	pix_scale = 0.147
	tol = 0.6
	min_dist = 2.0
	min_sep = 15.0
	min_prom = 0.0
	


	for infile in files:
		if infile.endswith('.tif'):
			## create a directory to save plots based on filename
			savepath = (os.path.splitext(infile))[0]
			savepath = indir + '/' + savepath
			if not os.path.exists(savepath):
				os.mkdir(savepath)

			savename = os.path.basename(os.path.normpath(savepath))

			## read in the tif
			imgstack = tf.imread(indir + '/' + infile).astype('float64')
			ts, ys, xs = imgstack.shape

			## small objects are anything smaller than 5000 pixels
			small_object = 5000

			## create an empty list to populate all distances for all times
			all_dists = []
			all_indices = []
			all_areas = []


			## TODO (check that we still need to do this) ignoring the last time point for now
			for i in range(ts - 1):
				print('Time point:', i)

				img = util.invert(imgstack[i, :, :])
				binary = process_img(img)

				## try removing small objects here from binary before calculating the 
				##	convex hull to prevent small objects from acting as vertices
				binary = remove_small_objects(binary, small_object, in_place = True)
				chull, verts, cx, cy = get_contour_hull(binary, ys, xs)

				## plot the chull rq to remind me what it is
				## we want the difference between the chull and shape defined by 
				##	cx, cy. It turns out that the convex hull uses the small objects
				##	we need to filter those out so that the convex hull only uses the largest
				##	object
				cont_hull = binary_fill_holes(binary)
				diff = cont_hull < chull
				area_img = np.zeros((ys, xs))

				## come up with a standardized area that coincides with furrows
				##	assume an equilateral triangle
				diff = remove_small_objects(diff, 1)	
				labels = label(diff)
				props = regionprops_table(labels, properties = ['area', 'coords'])	
				areas = props['area']
				coords = props['coords']	
				#print('coords:', coords)
				## TODO match areas with defects somehow	

				## make a closed loop we need the first coords at the 
				##	beginning and the end
				verts = np.append(verts, [verts[0]], axis = 0)
				cx = np.append(cx, cx[0])
				cy = np.append(cy, cy[0])
					
				## find the common set of coordinates between the contour and 
				##	convex hull vertices as a way to loop around the contours
				##	to find the defects
				c_verts, ind = find_common_coords(verts, list(zip(cx, cy)))
				c_verts = np.array(list(map(list, zip(*c_verts))))
				
				## create an empty list to populate with the distances found for each time point
				max_dists = []
				max_contx = []
				max_conty = []
				max_areas = []
				max_indices = []
				## loop around the hull vertices
				for j in range(len(c_verts[0, :]) - 1):
					#print('Currently on vertex:', j, j + 1)
					## grab the sections of contour points between hull vertices
					current_contx = cx[ind[j]:ind[j + 1]]
					current_conty = cy[ind[j]:ind[j + 1]]
					## define the orthogonal line that makes the section of the hull
					x0, x1 = c_verts[0, j], c_verts[0, j + 1]
					y0, y1 = c_verts[1, j], c_verts[1, j + 1]

					
					p1 = np.array([y0, x0])
					p2 = np.array([y1, x1])
					
					#fig, ax = plt.subplots()
					max_d = []
					max_i = [np.nan]	
					
					## experiment here - the distances are the rotated f(x)
					## values for a gradient calculation
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
	#					## this for loop seems inefficient, it loops through coords as many times
						## 	as there are defects, but maybe this is unavoidable
					
						## for each defect distance, find which area it belongs to and keep that area
						##	I'll use max_i coords for this

	#*************************************


							for c, pairs in enumerate(coords):
								for yy, xx in pairs:
									if np.isclose(peaks_y[0], xx, atol = tol) and np.isclose(peaks_x[0], yy, atol = tol):
										max_areas.append(areas[c]*pix_scale*pix_scale)		
										max_indices.append(c)
	#**************************************					


					## uncomment here if needed
					#for k in range(len(current_contx)):
					#	p3 = np.array([current_conty[k], current_contx[k]])
					#	if k == 0:
					#		d_past = find_dist_from_line(p1, p2, p3)
					#	else:
					#		d = find_dist_from_line(p1, p2, p3)
					#		#print(d)
					#		dp = find_defects(d, d_past)
					#		da = find_min_defects(d, d_past)
					#		## if there is no change in the distance
					#		## and no defect was found
					#		## 	assign the past distance the current distance 
	#
	#						if not dp:
	#							d_past = d
	#							continue
	#						else:
	#							#dp_count += 1
	#							## this only filters for 1
	#						#	print('da, dp not NaN', dp, da)
	#							if dp > 3:
	#								
	#								max_d.append(dp*pix_scale)
	#								max_i.append(p3)
	#								max_conty.append(p3[0])
	#								max_contx.append(p3[1])
	#				
	#				if len(max_d) > 0:
	#				#if max_d > 0:
	#					#max_dists.append(max_d*pix_scale)
	#					#max_conty.append(max_i[0])
	#					#max_contx.append(max_i[1])
	#					max_dists = max_d
	#					#print(max_i)
	#					#kjkj
						#max_conty = max_i[0]
	#					#max_contx = max_i[1]
	#					#print('Max distance:', max_d)
	#					#print('Max distance position:', max_i)				
	#
	#					## filters the areas based on intersection with max_dists points, 
	#					##	p1, p2, p3
						#tol = 0.6
	#					## this for loop seems inefficient, it loops through coords as many times
						## 	as there are defects, but maybe this is unavoidable
					
						## for each defect distance, find which area it belongs to and keep that area
						##	I'll use max_i coords for this
						#for c, pairs in enumerate(coords):
						#	for yy, xx in pairs:
								#if np.isclose(max_conty, xx, atol = tol) and np.isclose(max_contx, yy, atol = tol):
						#		if np.isclose(max_i[0], xx, atol = tol) and np.isclose(max_i[1], yy, atol = tol):
						#			max_areas.append(areas[c]*0.147*0.147)		
						#			max_indices.append(c)
	#									
	#			print('distances:', max_dists)
	#			print('Area(s) of defect:', max_areas)
				## append all defect distances in time
				#print(max_dists)			

				all_dists.append(max_dists)
				all_areas.append(max_areas)	
				all_indices.append(max_indices)

				for a in coords[max_indices]:
					area_img[a[:, 0], a[:, 1]] = 255
				
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

				#plt.show()
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
				time_pts = range(1, len(num_dists) + 1)

			#dists = {i for i in all_dists}
			#dists_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dists.items()]), columns = range(ts))
			dists_df = pd.DataFrame(all_dists)
			dists_df = dists_df.T
			dists_df.to_csv(os.path.join(savepath, savename + '_lengths_per_time_point.csv'))

			areas_df = pd.DataFrame(all_areas)
			areas_df = areas_df.T
			areas_df.to_csv(os.path.join(savepath, savename + '_areas_per_time_point.csv'))

			rcParams['font.family'] = 'Times New Roman'
			rcParams['font.size'] = 16

			## plot the mean defect lengths, areas, number versus time
			fig, ax = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
			ax[0].plot(time_pts, mean_dists, 'ko', label = 'Mean')
			ax[0].fill_between(time_pts, mean_dists - std_dists, mean_dists + std_dists, color = 'black', alpha = 0.35, label = 'Std.')
			ax[1].plot(time_pts, num_dists, 'ko')
			ax[2].plot(time_pts, mean_areas, 'ko', label = 'Mean')
			ax[2].fill_between(time_pts, mean_areas - std_areas, mean_areas + std_areas, color = 'black', alpha = 0.35, label = 'Std.')
			ax[0].set_ylabel('Furrow Length ($\mu$m)')
			ax[1].set_ylabel('No. Furrows')
			ax[2].set_ylabel('Area ($\mu$m$^2$)')
			ax[1].set_xlabel('Time Index')
			plt.subplots_adjust(hspace = .0)
			ax[0].grid()
			ax[1].grid()
			ax[2].grid()
			ax[0].legend(loc = 'upper left')
			ax[0].yaxis.get_major_locator().set_params(integer = True)
			ax[1].yaxis.get_major_locator().set_params(integer = True)
			ax[2].yaxis.get_major_locator().set_params(integer = True)
			plt.savefig(os.path.join(savepath, savename + '_stats.png'))

main()
