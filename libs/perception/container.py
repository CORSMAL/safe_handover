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
import os
import shutil
import copy

import time

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision

# Computer Vision libs
from libs.perception.mvg.projection import *
from libs.perception.fillingnet import *
from libs.perception.detection.detection import *

from pdb import set_trace as bp

#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Empty, Pasta, Rice, Water
FILLING_DENSITIES = [0, 0.41, 0.85, 1]
FILLING_LEVEL = ['Empty', 'Half-full (50%)', 'Full (90%)', 'Opaque']
FILLING_TYPE = ['Empty', 'Pasta', 'Rice', 'Water', 'Opaque']

JOINT2FILLINGLEVELIDX = [0, 1, 2, 1, 2, 1, 2, 3]
JOINT2FILLINGTYPEIDX = [0, 1, 1, 2, 2, 3, 3, 4]

FILLING_CLASSES = {
  0 : np.array([[1, 0, 0, 0, 0, 0, 0, 0]]).transpose(), # Empty, empty
	1 : np.array([[0, 1, 0, 0, 0, 0, 0, 0]]).transpose(), # Half-full, pasta
	2 : np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).transpose(), # Full, pasta
	3 : np.array([[0, 0, 0, 1, 0, 0, 0, 0]]).transpose(), # Half-full, rice
	4 : np.array([[0, 0, 0, 0, 1, 0, 0, 0]]).transpose(), # Full, rice
	5 : np.array([[0, 0, 0, 0, 0, 1, 0, 0]]).transpose(), # Half-full, water
	6 : np.array([[0, 0, 0, 0, 0, 0, 1, 0]]).transpose(), # Full, water
	7 : np.array([[0, 0, 0, 0, 0, 0, 0, 1]]).transpose()  # Opaque
}

def GetFeasibleStatusMapping():
    W = np.zeros((20,8))
    
    W[0][0] = 1
    W[5][1] = 1
    W[6][2] = 1
    W[9][3] = 1
    W[10][4] = 1
    W[13][5] = 1
    W[14][6] = 1
    W[19][7] = 1

    return W

def GetTransitionMatrix():
    A = np.zeros((8,8))
    
    A[0][0] = A[1][0] = A[3][0] = A[5][0] = 0.25
    A[0][1] = A[1][1] = A[2][1] = 0.33
    A[1][2] = A[2][2] = 0.5
    A[0][3] = A[3][3] = A[4][3] = 0.33
    A[3][4] = A[4][4] = 0.5
    A[0][5] = A[5][5] = A[6][5] = 0.33
    A[5][6] = A[6][6] = 0.5
    A[7][7] = 1.
    
    return A


def compute2Dcentroid(_rgb, _objmask):
    img_gray = cv2.cvtColor(_rgb, cv2.COLOR_BGR2GRAY)

    objmask = copy.deepcopy(_objmask)
    # objmask = objmask.astype(np.uint8).squeeze()

    img_mask = img_gray
    img_mask[objmask == 0] = 0

    # Intensity centroid
    m00 = m01 = m10 = 0

    x = np.array([a for a in range(0,img_mask.shape[1])])
    y = np.array([a for a in range(0,img_mask.shape[0])])

    m00 = np.sum(img_mask)
    m01 = np.sum(np.matmul(y, img_mask))
    m10 = np.sum(np.matmul(x, img_mask.transpose()))

    try:
        centroid = np.round(np.array((m10/m00, m01/m00)))
        return centroid
    except:
        print('Centroid not found')
        return None

def compute2DcentroidContours(_objmask):

    objmask = copy.deepcopy(_objmask)

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

    ### Re-projection error verification
    imagePoints1, _ = cv2.projectPoints(point3d.reshape(-1)[:3], c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], np.array([0.,0.,0.,0.,0.]))
    imagePoints2, _ = cv2.projectPoints(point3d.reshape(-1)[:3], c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], np.array([0.,0.,0.,0.,0.]))

    reperr = np.sqrt((cv2.norm(point1, imagePoints1.squeeze().astype(np.float64)) + cv2.norm(point2, imagePoints2.squeeze().astype(np.float64))) / 2)

    if reperr > 5:
        return None, reperr
    else:
        return point3d.reshape(-1)[:3], reperr


class container:
    def __init__(self, args, ncams):     
        # Object location
        self.centroid = None    # Location in 3D

        self.ncams = ncams
        
        # Object dimensions
        self.wb = 0             # Width at the bottom of the container
        self.wm = 0             # Width at the bottom of the container
        self.wt = 0             # Width at the bottom of the container
        self.maxwidth = 0       # Maximum width of the container
        self.h = 0              # Height of the container

        # Object volume and mass
        self.volume = 0         # Volume of the container
        self.mass = 0

        # Object shape
        self.estHeights = None
        self.estRadius = None

        # Filling properties
        self.fillingLevel = -1
        self.fillingType = -1   # 0: empty, 1: pasta, 2: rice, 3: water, -1: not estimated
        self.fillingMass = -1

        self.weight = 0

        self.ncams = ncams

        self.angularStep = 0
        self.anglesrad = None

        cam_str_list = []
        for n in range(1,ncams+1):
            cam_str_list.append('cam{:d}'.format(n))
        
        self.fillingLevelEstimator = FillingNet(args)
        self.filling_cam = dict.fromkeys(cam_str_list)

        for n in range(1,ncams+1):
            self.filling_cam['cam{:d}'.format(n)] = {'flevel' : -1, 'ftype' : -1, 'fjoint' : -1, 'flevpreds' : None, 'ftypepreds' : None, 'fjointpreds' : None}

        self.zt1 = None
        self.zt = None
        


    # Get and store dimensions
    def ComputeDimensions(self):
        if self.estRadius.size > 0 and self.estHeights.size>0:
            self.wb = self.estRadius[0] * 2.* 1000
            self.wm = self.estRadius[int(self.estRadius.shape[0] / 2.0)] * 2.* 1000
            self.wt = self.estRadius[-1] * 2.* 1000
            self.maxwidth = np.max(self.estRadius) *2. * 1000
            self.h = (self.estHeights[-1] - self.estHeights[0]) *1000

        else:
            self.wb = -1
            self.wm = -1
            self.wt = -1
            self.maxwidth = -1
            self.h = -1


    def GetVolume(self):
        return self.volume

    
    def ComputeVolume(self, h_step):
        # if self.estRadius is None or self.estHeights is None or not self.estRadius or not self.estHeights:
        #     self.volume = -1
        #     return 
        if self.estRadius.size > 0 and self.estHeights.size>0:
            volume = 0
            for pos, h in enumerate(self.estHeights):
                volume += h_step*math.pi*(self.estRadius[pos]**2)

            print("Estimated volume is: "+str(volume*1000000))
            self.volume = volume

            # bp()
        else:
            self.volume = -1

    def ComputeFillingMass(self):
        # mass = fullness x capacity x densitiy(filling)
        
        filling_lev = self.fillingLevel
        filling_type = self.fillingType
        container_vol = self.volume
        
        if filling_lev == -1 or filling_type == -1 or container_vol == -1:
            self.fillingMass = -1
            
            return

        if filling_type == 4:
            self.fillingMass = -2
        else:
            density = FILLING_DENSITIES[filling_type]
            self.fillingMass = filling_lev/100. * container_vol * 1000000 * density

    def getCircumference3Dpoints(self, angles, radius, height, normalize_xy):
        p3d = []

        centroid = self.centroid

        for angle in angles:
            if normalize_xy:
                p3d.append(np.array((radius*math.cos(angle), radius*math.sin(angle), height)).reshape(1,3))
            else:
                p3d.append(np.array((centroid[0]+(radius*math.cos(angle)), centroid[1]+(radius*math.sin(angle)), centroid[2]+height)).reshape(1,3))

        p3d = np.array(p3d)

        return p3d

    def computeAnglesRad(self):
        angles = np.linspace(0., 359., num=int((359.)/self.angularStep), endpoint=False)

        anglesrad = []
        for angle_d in angles:
            anglesrad.append(math.radians(angle_d))

        self.anglesrad = anglesrad
    
    def ShapeFitting(self, _c1, _c2, _objmask1, _objmask2, h_step, r_step):

        c1 = copy.deepcopy(_c1)
        c2 = copy.deepcopy(_c2)

        objmask1 = copy.deepcopy(_objmask1)
        objmask2 = copy.deepcopy(_objmask2)

        centroid = self.centroid

        # Radiuses
        step = 0.005                # meters
        minDiameter = 0.005         # meters
        maxDiameter = 0.15          # meters
        # angularStep = r_step        # degrees

        radiuses = np.linspace(maxDiameter/2, minDiameter/2, num=int((maxDiameter-minDiameter)/step))    
        # angles = np.linspace(0., 359., num=int((359.)/angularStep))

        # Heights
        step = h_step       # meters
        minHeight = -0.20   # meters
        maxHeight = 0.20     # meters
        
        heights = np.linspace(minHeight, maxHeight, num=int((maxHeight-minHeight)/step))

        # anglesrad = []
        # for angle_d in angles:
        #     anglesrad.append(math.radians(angle_d))
        if self.angularStep != r_step or self.anglesrad is None:
            self.angularStep = r_step
            self.computeAnglesRad()

        estRadius = []
        converged = []    
        
        for height in heights:
            converged_circ = False
            rad = radiuses[-1]

            for j in range(0, len(radiuses)):
                # bp()
                kk = j 
                
                if converged:
                    rad_conv = estRadius[-1]

                    kk = np.where(radiuses == rad_conv)[0][0]

                if j < max(kk-2,0):                   
                    continue

                rad = radiuses[j]
            
                # bp()

                # Sample 3D circumference
                p3d = self.getCircumference3Dpoints(self.anglesrad, rad, height, False)

                # Reproject to C1
                p2d_c1, _ = cv2.projectPoints(p3d, c1.extrinsic['rgb']['rvec'], c1.extrinsic['rgb']['tvec'], c1.intrinsic['rgb'], np.array([0.,0.,0.,0.,0.]))
                p2d_c1 = p2d_c1.squeeze().astype(int)
                # p2d_c1[:,0] -= offset

                if (p2d_c1[:,0] >= objmask1.shape[1]).any() or (p2d_c1[:,0] < 0).any() or (p2d_c1[:,1] >= objmask1.shape[0]).any() or (p2d_c1[:,1] < 0).any():
                    continue
                
                # Reproject to C2
                p2d_c2, _ = cv2.projectPoints(p3d, c2.extrinsic['rgb']['rvec'], c2.extrinsic['rgb']['tvec'], c2.intrinsic['rgb'], np.array([0.,0.,0.,0.,0.]))
                p2d_c2 = p2d_c2.squeeze().astype(int)
                # p2d_c2[:,0] -= offset

                if (p2d_c2[:,0] >= objmask2.shape[1]).any() or (p2d_c2[:,0] < 0).any() or (p2d_c2[:,1] >= objmask2.shape[0]).any() or (p2d_c2[:,1] < 0).any():
                    continue
                
                # Check if imaged points are in the segmentation
                areIn_c1 = objmask1[p2d_c1[:,1], p2d_c1[:,0]]
                areIn_c2 = objmask2[p2d_c2[:,1], p2d_c2[:,0]]

                if (np.count_nonzero(areIn_c1) == areIn_c1.shape[0]) and (np.count_nonzero(areIn_c2) == areIn_c2.shape[0]):
                    converged_circ = True
                    break
                
                if rad==minDiameter/2:
                    break


            estRadius.append(rad)
            converged.append(converged_circ)

        estRadius = np.array(estRadius)
        converged = np.array(converged)
       
        convRadius = estRadius[converged]
        convHeights = heights[converged]

        self.estRadius = convRadius[round(0.1*len(convRadius)):round(0.9*len(convRadius))]
        self.estHeights = convHeights[round(0.1*len(convHeights)):round(0.9*len(convHeights))]


    # Localisation and dimension estimation via shape reconstruction in 3D
    # Output: 
    #   - success_code: 
    #       1: success
    #      -1: no instances in either of the two views
    #      -2: large re-projection error
    def LoDE(self, rgb1, rgb2, c1, c2, instances1, instances2, h_step, r_step):
        print('LoDE')
        start_time = time.time()

        if instances1 is None or instances2 is None:
        	return -1

        # Add offset if image was cropped to be squared
        # centroid1[0] += self.offset
        # centroid2[0] += self.offset

        # Compute 2D centroid from the object mask in each view
        for j in range(0,len(instances1['masks'])):
            if torch.is_tensor(instances1['masks'][j]):
                seg = np.uint8(255.* (instances1['masks'][j] >= 0.5).detach().cpu().numpy().squeeze())
            else:
                seg = np.uint8(255.* (instances1['masks'][j] >= 0.5))
            instances1['centroid'].append(compute2Dcentroid(rgb1, seg))

        for j in range(0,len(instances2['masks'])):
            if torch.is_tensor(instances2['masks'][j]):
                seg = np.uint8(255.* (instances2['masks'][j] >= 0.5).detach().cpu().numpy().squeeze())
            else:
                seg = np.uint8(255.* (instances2['masks'][j] >= 0.5))
            instances2['centroid'].append(compute2Dcentroid(rgb2, seg))

        best_reperr = 10
        best_j = 0
        best_k = 0
        best_centroid = None
        for j in range(0,len(instances1['masks'])):
        	for k in range(0,len(instances2['masks'])):
        		# Compute centroid in 3D
        		centroid, reperr = compute3Dcentroid2views(c1, c2, instances1['centroid'][j], instances2['centroid'][k])

        		if reperr < best_reperr:
        			best_reperr = reperr
        			best_j = j
        			best_k = k
        			best_centroid = centroid

        if best_centroid is None:
        	print('3D localisation failure! Re-projection error: {:.3f} px'.format(best_reperr))
        	return -2

        self.centroid = best_centroid

        if torch.is_tensor(instances1['masks'][best_j]):
            objmask1 = np.uint8(255.* (instances1['masks'][best_j] >= 0.5).detach().cpu().numpy().squeeze())
            objmask2 = np.uint8(255.* (instances2['masks'][best_k] >= 0.5).detach().cpu().numpy().squeeze())
        else:
            objmask1 = np.uint8(255.* (instances1['masks'][j] >= 0.5))
            objmask2 = np.uint8(255.* (instances2['masks'][j] >= 0.5))

        # print('ShapeFitting')
        self.ShapeFitting(c1, c2, objmask1, objmask2, h_step, r_step)

        # print('ComputeDimensions')
        # Compute container dimensions
        self.ComputeDimensions()

        # print('ComputeVolume')
        # Compute volume of the container
        self.ComputeVolume(h_step)

        end_time = time.time()

        print("LoDE running time (milliseconds):", (end_time-start_time)*1000)

        return 1

    def printFillingLevel(self):
        if self.fillingLevel >=0 and self.fillingLevel < 4:
            print('Filling level: ' + FILLING_LEVEL[self.fillingLevel])
        else:
            print('Filling level: not estimated!')

    def printFillingType(self):
        if self.fillingType >=0 and self.fillingType < 5:
            print('Filling type: ' + FILLING_TYPE[self.fillingType])
        else:
            print('Filling type: not estimated!')

    def estimateFillingPropertiesPerCam(self, _img, _objmask, camID, args):
        start_time = time.time()

        img = copy.deepcopy(_img)   
        objmask = copy.deepcopy(_objmask)
        bbox = ComputeBboxFromMask(objmask)

        self.fillingLevelEstimator.run(img, bbox)

        self.filling_cam['cam{:d}'.format(camID)]['flevel'] = self.fillingLevelEstimator.getFillingLevel()
        self.filling_cam['cam{:d}'.format(camID)]['ftype'] = self.fillingLevelEstimator.getFillingType()
        self.filling_cam['cam{:d}'.format(camID)]['fjoint'] = self.fillingLevelEstimator.getJointFillingTypeLevel()
        self.filling_cam['cam{:d}'.format(camID)]['flevpreds'] = self.fillingLevelEstimator.getFillingLevelPredictions()
        self.filling_cam['cam{:d}'.format(camID)]['ftypepreds'] = self.fillingLevelEstimator.getFillingTypePredictions()
        self.filling_cam['cam{:d}'.format(camID)]['fjointpreds'] = self.fillingLevelEstimator.getJointFillingTypeLevelPredictions()

        end_time = time.time()
        print("Filling properties/cam running time (milliseconds):", (end_time-start_time)*1000)

        return self.fillingLevelEstimator.getFillingLevel(), self.fillingLevelEstimator.getFillingType(), bbox
        

    def estimateFillingProperties(self, frame_id):
        # print('Filling properties')

        start_time = time.time()

        # x1n = self.flevseq['cam1'][:]
        # x2n = self.flevseq['cam2'][:]

        # y1n = self.ftypseq['cam1'][:]
        # y2n = self.ftypseq['cam2'][:]

       
        # Xt = np.reshape(np.multiply(np.outer(x1n,y1n),np.outer(x2n,y2n)).transpose(), (1,20),'C')

        # W = GetFeasibleStatusMapping()
        
        A = GetTransitionMatrix()

        O = np.array([[1,1,1,1,1,1,1,1]])
        for n in range(1, self.ncams+1):
            O = np.multiply(O, self.filling_cam['cam{:d}'.format(n)]['fjointpreds'])

    
        # multiply: element-wise multiplication
        if frame_id > 0:           
            self.zt1 = self.zt
            pz = np.multiply(np.matmul(A.transpose(), self.zt1), O.transpose())
        else:
            self.zt1 = np.array([[0.5,0,0,0,0,0,0,0.5]]).transpose()
            pz = np.multiply(self.zt1, O.transpose())
        
        idx = np.argmax(pz)
        self.zt = FILLING_CLASSES[idx]

        # self.zt = self.zt / np.linalg.norm(self.zt)
        self.fillingLevel = JOINT2FILLINGLEVELIDX[idx]
        self.fillingType = JOINT2FILLINGTYPEIDX[idx]

        self.printFillingLevel()
        self.printFillingType()

        self.ComputeFillingMass()

        end_time = time.time()
        print('Filling properties running time (milliseconds)', (end_time-start_time)*1000)

    # Draw centroid
    def drawCentroid2D(self, cam, _img):
        img = copy.deepcopy(_img)
        
        # Draw 3D centroid projected to image
        point, _ = cv2.projectPoints(self.centroid, cam.extrinsic['rgb']['rvec'], cam.extrinsic['rgb']['tvec'], cam.intrinsic['rgb'], cam.distCoeffs)
        point = point.squeeze().astype(int)
        
        cv2.circle(img, tuple(point), 6, (128,0,0), -1)
        

    def saveResults(self, ofile, frame_cnt):
        if self.centroid is None:
            ofile.write('{:03d}, {:.6f}, {:.6f}, {:.6f}, '.format(frame_cnt, -1, -1, -1))
            ofile.write('{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, '.format(-1, -1, -1, -1, -1, -1))
        else:
            ofile.write('{:03d}, {:.6f}, {:.6f}, {:.6f}, '.format(frame_cnt, self.centroid[0], self.centroid[1], self.centroid[2]))
            ofile.write('{:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, '.format(self.wb, self.wm, self.wt, self.maxwidth, self.h, self.volume*1000000))
        
        ofile.write('{:d}, {:d}, {:4.2f}, {:4.2f}\n'.format(self.fillingLevel, self.fillingType, self.fillingMass, self.weight))


    def exportShapePointCloud(self, pointcloud_path):
        pointcloud = []

        for i, height in enumerate(self.estHeights):
            # Sample 3D circumference
            p3d = self.getCircumference3Dpoints(self.anglesrad, self.estRadius[i], height, True)
            pointcloud = pointcloud + p3d.tolist()

        np.savetxt(pointcloud_path, np.squeeze(np.array(pointcloud)), delimiter=",")

    def saveWidth(self, ofile, frame_cnt):
        if frame_cnt == 0:
            for i, height in enumerate(self.estHeights):
                ofile.write('{:03d}, {:.6f}, {:.6f}\n '.format(frame_cnt, height, self.estRadius[i]))
        ofile.close()


#####################################################
## OLD FUNCTIONS WITH MEDIAN
# def estimateFillingPropertiesPerCam(self, _img, _objmask, camID, args):

#     img = copy.deepcopy(_img)
#     objmask = copy.deepcopy(_objmask)

#     bbox = ComputeBboxFromMask(objmask)

#     fillingLevelEstimator = FillingNet(args)      
#     fillingLevel, fillingType = fillingLevelEstimator.run(img, bbox)

#     self.flevseq['cam{:d}'.format(camID)].append(fillingLevel)
#     self.ftypseq['cam{:d}'.format(camID)].append(fillingType)

#     return fillingLevel, fillingType, bbox
    

# def estimateFillingProperties(self, frame_id, wnd_sz):
#     if frame_id < wnd_sz:
#         L1 = np.array(self.flevseq['cam1'][:])
#         L2 = np.array(self.flevseq['cam2'][:])

#         T1 = np.array(self.ftypseq['cam1'][:])
#         T2 = np.array(self.ftypseq['cam2'][:])
#     else:
#         bp()
#         L1 = np.array(self.flevseq['cam1'][-wnd_sz:])
#         L2 = np.array(self.flevseq['cam2'][-wnd_sz:])

#         T1 = np.array(self.ftypseq['cam1'][-wnd_sz:])
#         T2 = np.array(self.ftypseq['cam2'][-wnd_sz:])

#     self.fillingLevel = int(np.round(np.median(np.concatenate((L1,L2,np.array([self.fillingLevel]))))))
#     self.fillingType = int(np.round(np.median(np.concatenate((T1,T2,np.array([self.fillingType]))))))

#     self.printFillingLevel()
#     self.printFillingType()

#     self.ComputeFillingMass()
