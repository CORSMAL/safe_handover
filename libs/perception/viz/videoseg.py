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

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision

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
	  fl_str = 'empty'
	elif fillingLevel == 1:
		fl_str = 'half-full'
	elif fillingLevel == 2:
	  fl_str = 'full'
	elif fillingLevel == 3:
		fl_str = 'opaque'
	
	return fl_str

def makevideoseg(args):
	try:
		vidcap = cv2.VideoCapture(args.videopath + '/rgb/' + args.videoname + '.mp4')

		if not os.path.exists(args.videopath + '/rgbmask/'):
			os.makedirs(args.videopath + '/rgbmask')
			os.makedirs(args.videopath + '/rgbmask/tmp')
		elif not os.path.exists(args.videopath + '/rgbmask/tmp'):
			os.makedirs(args.videopath + '/rgbmask/tmp')

		L = np.loadtxt(args.videopath + '/volume_estimation/' + args.videoname[:-3] + '_fillinglevel.txt', dtype='int', delimiter=',')

		fr=0 # Frame counter

		# Read first frame
		print('Frame #' + str(fr))
		success, img = readNextImage(vidcap)
		seg = cv2.imread(args.videopath + '/seg_mask/' + args.videoname +'/{:04d}.png'.format(fr))

		# img[:, :, 0] = (seg[:,:,0] > 0) * 255 + (seg[:,:,0] == 0) * img[:, :, 0]*0.65
		img[:, :, 1] = (seg[:,:,1] > 0) * 255 + (seg[:,:,1] == 0) * img[:, :, 1]*0.65
		# img[:, :, 2] = (seg[:,:,2] > 0) * 255 + (seg[:,:,2] == 0) * img[:, :, 2]*0.65	
		img[:, :, 0] = img[:, :, 0]*0.65
		img[:, :, 2] = img[:, :, 2]*0.65

		if args.cam_id == 1:
			fl = L[fr,1]
		else:
			fl = L[fr,2]

		fl_str = getFillingLevelString(fl)

		cv2.putText(img, fl_str, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), lineType=cv2.LINE_AA) 

		cv2.imwrite(args.videopath +'/rgbmask/tmp/'+ '{:04d}.png'.format(fr), img)

		fr += 1

		while success:
			print("Frame #" + str(fr))
			success, img = readNextImage(vidcap)		
			seg = cv2.imread(args.videopath + '/seg_mask/' + args.videoname +'/{:04d}.png'.format(fr))

			# img[:, :, 0] = (seg[:,:,0] > 0) * 255 + (seg[:,:,0] == 0) * img[:, :, 0]*0.65
			img[:, :, 1] = (seg[:,:,1] > 0) * 255 + (seg[:,:,1] == 0) * img[:, :, 1]*0.65
			# img[:, :, 2] = (seg[:,:,2] > 0) * 255 + (seg[:,:,2] == 0) * img[:, :, 2]*0.65	
			img[:, :, 0] = img[:, :, 0]*0.65
			img[:, :, 2] = img[:, :, 2]*0.65	
			
			if args.cam_id == 1:
				fl = L[fr,1]
			else:
				fl = L[fr,2]

			fl_str = getFillingLevelString(fl)

			cv2.putText(img, fl_str, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), lineType=cv2.LINE_AA) 
			
			cv2.imwrite(args.videopath +'/rgbmask/tmp/'+ '{:04d}.png'.format(fr), img)

			fr += 1

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
	# parser.add_argument('--respath', type=str)
	parser.add_argument('--cam_id', type=int)

	args = parser.parse_args()

	makevideoseg(args)