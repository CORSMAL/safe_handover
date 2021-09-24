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

class Camera:
	def __init__(self, args):
		
    self.camId = camId

		self.intrinsic = dict.fromkeys(['rgb'])
		self.extrinsic = dict.fromkeys(['rgb'])
		self.distCoeffs = None

		self.extrinsic['rgb'] = dict.fromkeys(['rvec','tvec','projMatrix'])

		self.image = dict.fromkeys(['rgb'])

		self.track = None
		self.frame_iter = 0
		self.state = "wait"
		
		self.topic = "/camera1/color/image_raw"
    self.pub_topic = "/siammask1/image"
    self.ROI = None
    self.seg = None
    self.points = None

    self.cvBridge = CvBridge()

  def cam_callback(self, data):    
		self.image['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

		if self.frame_iter == 0:
		    try:          
		        # Calibration
		        if self.args.load_recorded_video:
		            self.distCoeffs = np.zeros((1,5), dtype=np.float64)

		            with open(self.args.calib_1, 'rb') as f:
		                u = pickle._Unpickler(f)
		                u.encoding = 'latin1'
		                calibration = u.load()
		            
		            self.intrinsic['rgb'] = calibration[0]['rgb']
		            self.extrinsic['rgb'] = calibration[1]['rgb']
		            
		        else:
		            # self.cam.cameraPose(self.image)

		        # Detection
		        rospy.loginfo('Detecting object from camera 1...')
		        img_square, self.offset = prepareImage(self.image['rgb'])
		        output = self.detectionModel([trf(img_square).cuda()])

		        self.ROI, self.seg, self.points, self.img2plotc = postProcessingDetection(self.camId, img_square, output[0], draw=False)
		        if self.args.save_first_frame:
		            cv2.imwrite(self.savedir1+'/first_frame/'+self.fname1+'.png', img_square)

		        if self.ROI is not None:
		            # Move all to full image. To remove when bug about make it square is solved
		            self.ROI[0,0] += self.offset
		            self.ROI[0,2] += self.offset
		            self.points[:,0] += self.offset

		            # Tracking
		            rospy.loginfo('Initialising tracking from camera 1...')
		            self.track = maskTrack(self.ROI)
		            self.track.run(self.image['rgb'], 'init')

		            self.state = "ready"
		            rospy.loginfo('Camera 1 ready!')
		        
		        # Save segmentation mask
		        if self.args.save_seg_mask:
		            mask = self.seg.astype(np.uint8).squeeze()
		            img_square[:, :, 1] = (mask > 0) * 255 + (mask == 0) * img_square[:, :, 1]
		            cv2.imwrite(self.savedir1+'/seg_mask/'+self.fname1+'/{:04d}.png'.format(self.frame_iter), mask)

		        else:
		            return
		    except:
		        return

		self.frame_iter += 1


	def readNextImage(self, vidcap):
		success, img = vidcap.read()

		if not success:
		  print('Finished reading image from video of camera {:d}.'.format(self.camId))
		  img_msg = None
		else:
			img_msg = self.cvBridge.cv2_to_imgmsg(img, encoding="bgr8")
			self.cam_callback(data=img_msg)

		return success, img_msg