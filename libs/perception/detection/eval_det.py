#!/usr/bin/env python
#
#--------------------------------------------------------------------------------
# Authors: 
#   - Yik Lung Pang: y.l.pang@qmul.ac.uk
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
# MIT License

# Copyright (c) 2021 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#--------------------------------------------------------------------------------

#

# System libs
import glob
import sys
import argparse
import shutil
import os
import copy

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision

from detection import *

from pdb import set_trace as bp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


fnames_filling =  ['fi0_fu0','fi1_fu1', 'fi1_fu2','fi2_fu1', 'fi2_fu2', 'fi3_fu1', 'fi3_fu2']
fnames_filling2 = ['fi0_fu0','fi1_fu1', 'fi1_fu2','fi2_fu1', 'fi2_fu2']

def populate_filenames(mode):
	list_filenames = []
	for s in range(0,3):
		str_s = 's{:d}_'.format(s)
		
		for b in range(0,2):
			str_b = '_b{:d}_'.format(b)
			
			for l in range(0,2):
				str_l = 'l{:d}'.format(l)

				if mode == 0:
					for f in fnames_filling:
						list_filenames.append(str_s + f + str_b + str_l)
				else:
					for f in fnames_filling2:
						list_filenames.append(str_s + f + str_b + str_l)

	return list_filenames


def resizeImage(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	return resized 


def detector(args):
	try:
		detectionModel = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)
		detectionModel.load_state_dict(torch.load('../../data/models/coco_maskrcnn_resnet50_fpn_2cat.pth', map_location='cpu'))
		detectionModel.eval()
		detectionModel.cuda()
	except Exception as e:
		print('Error in loading the model')
		return	

	ofile_c1 = open('detections_c1.txt', 'w')
	ofile_c2 = open('detections_c2.txt', 'w')

	ofile_c1.write('Filename, Success, bbox_x1, bbox_y1, bbox_x2, bbox_y2, score\n')
	ofile_c2.write('Filename, Success, bbox_x1, bbox_y1, bbox_x2, bbox_y2, score\n')

	for c in range(1,10):
		containerpath = args.datapath + '/{:d}'.format(c)
		
		if not os.path.exists(containerpath + '/maskrcnn/'):
			os.makedirs(containerpath + '/maskrcnn')

		if c < 7:
			list_files = populate_filenames(0)
		else:
			list_files = populate_filenames(1)

		for f in list_files:
			for cam_id in range(1,3):
				videopath = containerpath + '/rgb/' + f + '_c{:d}.mp4'.format(cam_id)
				
				print(videopath)
				try:
					vidcap = cv2.VideoCapture(videopath)
				except Exception as e:
					print('Error in opening video file')

					if cam_id == 1:
						ofile_c1.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(-1, 0, 0, 0, 0, -1))
					elif cam_id == 2:
						ofile_c2.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(-1, 0, 0, 0, 0, -1))

					continue

				# fr=0 # Frame counter
				# print("Frame #" + str(fr))

				success, img = vidcap.read()
				if not success:
					print('Finished reading image from video of camera.')
					if cam_id == 1:
						ofile_c1.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(-1, 0, 0, 0, 0, -1))
					elif cam_id == 2:
						ofile_c2.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(-1, 0, 0, 0, 0, -1))
				else:
					det_succ, ROI, score, seg, points, img2plot = instanceSegmentationMaskRCNN(cam_id, img, detectionModel, draw=True)

					if det_succ:
						resized_img = resizeImage(img2plot, 50)
						
						cv2.imwrite(containerpath + '/maskrcnn/{:d}_'.format(c) + f + '_c{:d}.png'.format(cam_id), resized_img)

						if cam_id == 1:
							ofile_c1.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(int(det_succ), ROI[0,0], ROI[0,1], ROI[0,2], ROI[0,3], score))
						elif cam_id == 2:
							ofile_c2.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(int(det_succ), ROI[0,0], ROI[0,1], ROI[0,2], ROI[0,3], score))
					else:
						print('No detections')
						if cam_id == 1:
							ofile_c1.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(0, 0, 0, 0, 0, -1))
						elif cam_id == 2:
							ofile_c2.write('{:d}_'.format(c) + f + ', {:d}, {:4d}, {:4d}, {:4d}, {:4d}, {:.2f}\n'.format(0, 0, 0, 0, 0, -1))

	ofile_c1.close()
	ofile_c2.close()

# Compute Localisation Success Ratio (LSR)
# LSR: number of configuration with a detection over the total number of configurations
# Analysis by scenario and container ID
def ComputeLSR(filename):
	scenarios = {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }
	
	containers = { 'container1' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 },
								 'container2' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container3' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container4' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container5' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container6' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container7' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container8' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }, 
								 'container9' : {'scenario1' : {'detections': [], 'LSR': 0}, 
							 'scenario2' : {'detections': [], 'LSR': 0}, 
							 'scenario3' : {'detections': [], 'LSR': 0}
							 }
								}

	print(filename)	
	filein = open(filename, 'r')
	cnt = 0
	for line in filein:
		# print(str(cnt) + ': ' + line )
		if cnt > 0:
			myline_s = line.split(',')
			sc = myline_s[0].split('_')

			x = containers['container' + sc[0]]['scenario' + str(int(sc[1][-1]) + 1)]['detections']
			# print(x)
			x.append(int(myline_s[1]))
			# print(x)
			containers['container' + sc[0]]['scenario' + str(int(sc[1][-1]) + 1)]['detections'] = x
			# print(containers['container' + sc[0]]['scenario' + str(int(sc[1][-1]) + 1)]['detections'])

		cnt += 1

	# bp()

	res = np.zeros((3,9))
	tot = np.zeros((3,9))
	nconf = np.zeros((3,9))
	for c in range(1,10):
		for s in range(1,4):
			print('container' + str(c) + '\tscenario' + str(s))
			tmparray = np.array(containers['container' + str(c)]['scenario' + str(s)]['detections'])

			print(tmparray)
			
			containers['container' + str(c)]['scenario' + str(s)]['LSR'] = tmparray.sum() / tmparray.size
			
			res[s-1,c-1] = tmparray.sum() / tmparray.size
			tot[s-1,c-1] = tmparray.sum() 
			nconf[s-1,c-1] = tmparray.size 

	print(res * 100)
	print(tot)
	print(nconf)
	bp()

	print(np.sum(tot, axis=0) / np.sum(nconf, axis=0)) 
	print(np.sum(tot, axis=1) / np.sum(nconf, axis=1))


if __name__ == '__main__':
	print('Initialising:')
	print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
	print('OpenCV {}'.format(cv2.__version__))
	print('PyTorch {}'.format(torch.__version__))
	print('Torchvision {}'.format(torchvision.__version__))

	if device == 'cuda':
		torch.cuda.set_device(0)
	print('Using {}'.format(device))

	# Parse arguments
	parser = argparse.ArgumentParser()
	# parser.add_argument('--load_recorded_video', type=int, choices=[0,1], default=0)
	parser.add_argument('--datapath', type=str)
	# parser.add_argument('--videoname', type=str)
	# parser.add_argument('--filling_lev', type=str)
	# parser.add_argument('--respath', type=str)
	# parser.add_argument('--cam_id', type=int)

	args = parser.parse_args()

	detector(args)

	# ComputeLSR('detections_c1.txt')