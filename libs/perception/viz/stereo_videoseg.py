#!/usr/bin/env python

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

from pdb import set_trace as bp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def readNextImage(vidcap):
  success, img = vidcap.read()
  # img_msg = None

  if not success:
      print('Finished reading image from video of camera.')
  # else:
  #     img_msg = self.cvBridge.cv2_to_imgmsg(img1, encoding="bgr8")
  #     self.cam1_callback(data=img_msg1)

  return success, img

def getFillingLevelString(fillingLevel):
	fl_str = 'not estimated'

	if fillingLevel == 0:
	  fl_str = 'Level: empty'
	elif fillingLevel == 1:
		fl_str = 'Level: half-full'
	elif fillingLevel == 2:
	  fl_str = 'Level: full'
	elif fillingLevel == 3:
		fl_str = 'Level: opaque'
	
	return fl_str

def getFillingTypeString(fillingType):
	fl_str = 'not estimated'

	if fillingType == 0:
	  fl_str = 'Type: empty'
	elif fillingType == 1:
		fl_str = 'Type: pasta'
	elif fillingType == 2:
	  fl_str = 'Type: rice'
	elif fillingType == 3:
		fl_str = 'Type: water'
	elif fillingType == 4:
		fl_str = 'Type: opaque'
	
	return fl_str


def resizeImage(img, scale_percent):
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	return resized 


def getImageCam(args, vidcap, fr, L, cam_id):
	success, img = readNextImage(vidcap)

	if success:
		# print(args.videopath + '/seg_mask/' + args.videoname + '_c{:d}/{:04d}.png'.format(cam_id,fr))
		seg = cv2.imread(args.respath + '/seg_mask/' + args.videoname + '_c{:d}/{:04d}.png'.format(cam_id,fr))

		if cam_id == 1:
			bbox = L[fr,1:5]
			fl = L[fr,5]
			ft = L[fr,6]
		else:
			bbox = L[fr,7:11]
			fl = L[fr,11]
			ft = L[fr,12]

		if all(bbox == -1):
			return success, img 

		ROI = copy.deepcopy(img[bbox[1]:bbox[3],bbox[0]:bbox[2],:])

		# img[:, :, 0] = (seg[:,:,0] > 0) * 255 + (seg[:,:,0] == 0) * img[:, :, 0]*0.65
		img[:, :, 1] = (seg[:,:,1] > 0) * 255 + (seg[:,:,1] == 0) * img[:, :, 1]*0.65
		# img[:, :, 2] = (seg[:,:,2] > 0) * 255 + (seg[:,:,2] == 0) * img[:, :, 2]*0.65	
		img[:, :, 0] = img[:, :, 0]*0.65
		img[:, :, 2] = img[:, :, 2]*0.65

		ROI = resizeImage(ROI, 200)
		img[150:ROI.shape[0]+150,-ROI.shape[1]-50:-50,:] = ROI

		fl_str = getFillingLevelString(fl)
		ft_str = getFillingTypeString(ft)

		cv2.putText(img, fl_str, (img.shape[1]-250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
		cv2.putText(img, ft_str, (img.shape[1]-250,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

	return success, img



def makevideoseg(args):
	try:
		# bp()
		vidcap1 = cv2.VideoCapture(args.videopath + '/rgb/' + args.videoname + '_c1.mp4')
		vidcap2 = cv2.VideoCapture(args.videopath + '/rgb/' + args.videoname + '_c2.mp4')

		if vidcap1.get(cv2.CAP_PROP_FRAME_COUNT) != vidcap2.get(cv2.CAP_PROP_FRAME_COUNT):
			print('Different number of frames')

		if not os.path.exists(args.respath + '/rgbmask/'):
			os.makedirs(args.respath + '/rgbmask')
			os.makedirs(args.respath + '/rgbmask/tmp')
		elif not os.path.exists(args.respath + '/rgbmask/tmp'):
			os.makedirs(args.respath + '/rgbmask/tmp')

		# L = np.loadtxt(args.videopath + '/volume_estimation/' + args.videoname + '_c1_fillinglevel.txt', dtype='int', delimiter=',')
		L = np.loadtxt(args.respath + '/volume_estimation/' + args.videoname + '_fillinglevel.txt', dtype='int', delimiter=',')
		est = np.loadtxt(args.respath + '/volume_estimation/' + args.videoname + '.txt', dtype='float', delimiter=',')

		succ1 = succ2 = True

		fr=0 # Frame counter
		
		if vidcap1.get(cv2.CAP_PROP_FRAME_COUNT) != est.shape[0]:
			print('Inconsistency')

		while succ1 and succ2:
			print("Frame #" + str(fr))

			# if fr == 33:
			# 	bp()

			succ1, img1 = getImageCam(args, vidcap1, fr, L, 1)
			succ2, img2 = getImageCam(args, vidcap2, fr, L, 2)

			if succ1 and succ2:
				resized_img1 = resizeImage(img1, 50)
				resized_img2 = resizeImage(img2, 50)

				img = np.concatenate((resized_img1, resized_img2), axis=1)
				# bp()

				fl_str = getFillingLevelString(est[fr, 10])
				ft_str = getFillingTypeString(est[fr, 11])
				
				cv2.putText(img, fl_str, (int(img.shape[1]/2-100),int(img.shape[0]-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, lineType=cv2.LINE_AA)
				cv2.putText(img, ft_str, (int(img.shape[1]/2-100),int(img.shape[0]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, lineType=cv2.LINE_AA)

				cv2.imwrite(args.respath +'/rgbmask/tmp/'+ '{:04d}.png'.format(fr), img)
				fr += 1
			else:
				break

	except Exception as e:
		print('Error')

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
	parser.add_argument('--videopath', type=str)
	parser.add_argument('--videoname', type=str)
	# parser.add_argument('--filling_lev', type=str)
	parser.add_argument('--respath', type=str)
	# parser.add_argument('--cam_id', type=int)

	args = parser.parse_args()

	print(args.respath)
	makevideoseg(args)