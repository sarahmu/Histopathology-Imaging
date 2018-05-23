import numpy as np
import openslide
from PIL import Image
from openslide import OpenSlideError
import math
import os
import multiprocessing as mp
import sys, getopt
from utils import local_clock
from utils import chunk

def get_white_trans_fracs(patch_np):
	"""
	Calculate the fractions of white background and transparent background

	Inputs:
	- patch_np: Numpy array of a patch in RGBA format

	Outputs:
	- frac_white: Float for the fraction of white background
	- frac_trans: Float for the fraction of transparent background
	"""
	white = np.mean(patch_np, 2) > 200
	transparent = np.mean(patch_np, 2) == 0
	num_pixels = patch_np.shape[0] * patch_np.shape[1]
	frac_white = np.sum(white) / num_pixels
	frac_trans = np.sum(transparent) / num_pixels
	return frac_white, frac_trans

def patch_slide(slide_file, color_dir, gray_dir, log_file, white_threshold=0.95, trans_threshold=0.1, patch_size=224, 
	limit=5000):
	"""
	Break a whole-slide image into patches. Save the original patches and the gray-scale patches with 
	useful content to different directories.

	Inputs:
	- slide_file: String for the file path to the whole-slide image
	- color_dir: String for the output directory of the color patches
	- gray_dir: String for the output directory of the gray patches
	- log_file: String for the log file to record progress
	- white_threshold: Float. Save a patch if the fraction of white background is less than this. 
	- trans_threshold: Float. Save a patch if the fraction of transparent background is less than this.
	- patch_size: Integer for the width and height of the patches
	- limit: Integer for the maximum number of patches to be generated

	Outputs: None. This method output patches in png files to the output directories.
	"""
	with open(log_file, 'a') as log_handle:
		log_handle.write(local_clock() + '  Start processing patches for ' + slide_file + '\n')
	# Parse the file name to get patient id and node id
	slide_filename = slide_file.split('/')[-1]
	patient_id = slide_filename.split('_')[1]
	node_id = slide_filename.split('_')[-1].split('.')[0]

	slide = openslide.open_slide(slide_file)
	w, h = slide.dimensions

	# Only consider the middle havles
	w_start = math.floor(w / 4)
	w_end = math.floor(w / 4 * 3)
	h_start = math.floor(h / 4)
	h_end = math.floor(h / 4 * 3)
	x_list = np.arange(start=w_start, stop=w_end, step=patch_size)
	y_list = np.arange(start=h_start, stop=h_end, step=patch_size)

	patch_idx = 0
	for x in x_list:
		for y in y_list:
			patch = slide.read_region((x,y), 0, (patch_size,patch_size))
			patch_np = np.array(patch, dtype='uint8')
			frac_white, frac_trans = get_white_trans_fracs(patch_np)
			if frac_white < white_threshold and frac_trans < trans_threshold:
				patch_name = patient_id + '_' + node_id + '_' + str(patch_idx) + '_' + str(x) + '_' + str(y) + '.png'
				gray_patch = patch.convert('L')
				patch.save(color_dir + patch_name)
				gray_patch.save(gray_dir + 'gray_' + patch_name)
				patch_idx += 1
			if patch_idx % 1000 == 0:
				with open(log_file, 'a') as log_handle:
					log_handle.write(local_clock() + '  Finished %d patches\n' % (patch_idx))
			if patch_idx == limit:
				break
		if patch_idx == limit:
			break
	with open(log_file, 'a') as log_handle:
		log_handle.write(local_clock() + '  Done with ' + slide_file + '!\n')
	pass

def build_worker_params_list(file_list, num_workers):
	file_chunks = chunk(file_list, num_workers)
	worker_params_list = []
	for i in range(num_workers):
		worker_params_list.append({'idx':i, 'slide_files':file_chunks[i]})
	return worker_params_list

if __name__ == '__main__':
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'd:c:g:w:t:p:l:n:o:', 
			['directory=', 'color_dir=', 'gray_dir=', 'white_threshold=', 'trans_threshold=', 'patch_size=', 'limit=', 'num_workers=', 'log_dir='])
	except getopt.GetoptError:
		print ('patch.py -d <directory> -c <color_dir> -g <gray_dir> -w <white_threshold> -t <trans_threshold> -p <patch_size> -l <limit> -n <num_workers> -o <log_dir>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-d','--directory'):
			directory = arg
		elif opt in ('-c', '--color_dir'):
			color_dir = arg
		elif opt in ('-g', '--gray_dir'):
			gray_dir = arg
		elif opt in ('-w', '--white_threshold'):
			white_threshold = float(arg)
		elif opt in ('-t', '--trans_threshold'):
			trans_threshold = float(arg)
		elif opt in ('-p', '--patch_size'):
			patch_size = int(arg)
		elif opt in ('-l', '--limit'):
			limit = int(arg)
		elif opt in ('-n', '--num_workers'):
			num_workers = int(arg)
		elif opt in ('-o', '--log_dir'):
			log_dir = arg
	print('Running patch.py with params: --directory= ' + directory + ', --color_dir=' + color_dir + ', --gray_dir=' + \
		gray_dir + ', --white_threshold=' + str(white_threshold) + ', --trans_threshold=' + str(trans_threshold) + \
		', --patch_size=' + str(patch_size) + ', --limit=' + str(limit) + ', --num_workers=' + str(num_workers) + \
		', --log_dir=' + log_dir)
	tif_files = [directory + file for file in os.listdir(directory) if file.endswith('.tif')]
	worker_params_list = build_worker_params_list(tif_files, num_workers)

	def patch_slides_wrapper(worker_params, color_dir=color_dir, gray_dir=gray_dir, white_threshold=white_threshold, 
		trans_threshold=trans_threshold, patch_size=patch_size, limit=limit, log_dir=log_dir):
		idx = worker_params['idx']
		slide_files = worker_params['slide_files']
		log_file = log_dir + 'log_' + str(idx) + '.txt'
		with open(log_file, 'w') as log_handle:
			log_handle.write(local_clock() + '  Start processing slides for worker ' + str(idx) + '\n')
		for slide_file in slide_files:
			patch_slide(slide_file=slide_file, color_dir=color_dir, gray_dir=gray_dir, log_file=log_file, 
				white_threshold=white_threshold, trans_threshold=trans_threshold, patch_size=patch_size, limit=limit)
		with open(log_file, 'a') as log_handle:
			log_handle.write(local_clock() + '  Done for worker ' + str(idx) + '!')

	pool = mp.Pool(processes=num_workers)
	pool.map(patch_slides_wrapper, worker_params_list)
	print('Done with patching!')