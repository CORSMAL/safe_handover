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
import torchvision.transforms as T

from numpy import linalg as LA

from cv_bridge import CvBridge, CvBridgeError


def compute2Dcentroid(objmask):
  objmask = objmask.copy()
  # objmask = objmask.astype(np.uint8).squeeze()

  # Get the largest contour
	# _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours, _ = cv2.findContours(objmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

	# Get centroid of the largest contour
	M = cv2.moments(largest_contour)

	try:
		centroid = np.array((M['m10']/M['m00'], M['m01']/M['m00']))
		return centroid
	except:
		print('Centroid not found')
		return None


def compute3Dcentroid2views(c1, c2, point1, point2, undistort=True):

	if (point1.dtype != 'float64'):
		point1 = point1.astype(np.float64)

	if (point2.dtype != 'float64'):
		point2 = point2.astype(np.float64)

	point3d = cv2.triangulatePoints(c1.extrinsic['rgb']['projMatrix'], c2.extrinsic['rgb']['projMatrix'], point1.reshape(2,1), point2.reshape(2,1)).transpose()
	for point in point3d:
		point /= point[-1]
	
	return point3d.reshape(-1)

def getObjectDimensions(self):
	centroid1 = getCentroid(self.c1, self.seg1)
	centroid1[0] += self.offset
	centroid2 = getCentroid(self.c2, self.seg2)
	centroid2[0] += self.offset

	centroid = cv2.triangulatePoints(self.c1.extrinsic['rgb']['projMatrix'], self.c2.extrinsic['rgb']['projMatrix'], centroid1, centroid2).transpose()
	centroid /= centroid[:,-1].reshape(-1,1)
	centroid = centroid[:,:-1].reshape(-1)

	# Save initial location
	self.initial_location = np.append(copy.deepcopy(centroid).squeeze(), 1.).reshape(4,1)
	if not(self.args.load_recorded_video):
	    self.initial_location = np.matmul(self.camera_robot_transformation, self.initial_location)        
	    self.initial_state.pose.position.x = self.initial_location[0]
	    self.initial_state.pose.position.y = self.initial_location[1]
	    self.initial_state.pose.position.z = self.initial_location[2]
	    self.initial_state.scale.x = 0
	    self.initial_state.scale.y = 0
	    self.initial_state.scale.z = 0

	if self.args.use_LoDE:
	    h, wm, v = getObjectDimensionsLODE(self.c1, self.c2, self.seg1, self.seg2, self.cam1['rgb'], self.cam2['rgb'], centroid, self.offset, self.args.LoDE_hstep, self.args.LoDE_rstep, self.args.export_pointcloud, self.pointcloud_path, draw=True)
	    # self.results_file.write('Estimated object dimensions: wm='+str(wm)+'mm, h='+str(h)+'mm, v='+str(v*1000000)+'mL')
	    self.results_file.write('# Width bottom (mm) | Width middle (mm) | Width top (mm) | Height (mm) | Volume (mL)\n')
	    self.results_file.write('{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(0,wm,0,h,v*1000000))

	else:
	    wb_c1, h_c1 = getObjectDimensions(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, 0, draw=False)
	    wb_c2, h_c2 = getObjectDimensions(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, 0, draw=False)
	    wm_c1, _ = getObjectDimensions(self.c1, self.seg1, self.cam1['rgb'], centroid, self.offset, centroid[2], draw=False)
	    wm_c2, _ = getObjectDimensions(self.c2, self.seg2, self.cam2['rgb'], centroid, self.offset, centroid[2], draw=False)

	    
	    wb = (wb_c1+wb_c2)/2
	    wm = (wm_c1+wm_c2)/2
	    wt = LA.norm(wb - wm) + wm
	    h = (h_c1+h_c2)/2

	    self.results_file.write('# Width bottom (mm) | Width middle (mm) | Width top (mm) | Height (mm) | Volume (mL)\n')
	    self.results_file.write('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(wb*1000, wm*1000, wt*1000, h*10000,0))
	    # self.results_file.write('Estimated object dimenions: w_t={:.0f}mm, w_b={:.0f}mm, h={:.0f}mm'.format(wt*1000, wb*1000, h*1000))

	self.glass.w = wm
	self.glass.h = h