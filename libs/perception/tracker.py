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

# Numeric libs
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from numpy import linalg as LA

# Computer Vision libs
from libs.perception.mvg.projection import *
from libs.perception.detection.detection import *
from libs.perception.siamMask.tools.siamMask import siamMask as maskTrack
from libs.perception.container import *

from pdb import set_trace as bp

from cv_bridge import CvBridge, CvBridgeError

import traceback


#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trf = T.Compose([T.ToPILImage(),
                 T.ToTensor()])


def readCalibration(calib, cam):
    cam.intrinsic = dict.fromkeys(['rgb'])
    cam.extrinsic = dict.fromkeys(['rgb'])

    cam.distCoeffs = np.zeros((1,5), dtype=np.float64)

    # calibration = pickle.load(open(self.args.calib_1, 'rb'))

    with open(calib, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        calibration = u.load()

    cam.intrinsic['rgb'] = calibration[0]['rgb']
    cam.extrinsic['rgb'] = calibration[1]['rgb']

    return cam


class Tracker:
    def __init__(self, args):

        # Create object class
        self.container = container(args, 2) # Initialise the number of cameras in the containers

        self.model = self.LoadModel(args)
        self.detection = Camera(args.detpretrained)


        self.args = args

        self.results_file = None
        self.results_file_width = None
        self.track_res_file = None
        self.pointcloud_path = None
        self.fout_res_lev = None

        self.fr = 0

        self.cvBridge = CvBridge()
        
        # Camera 1
        self.c1 = projection(camId=1)
        self.track1 = None
        self.fr1 = 0
        self.cam1_state = "wait"
        self.cam1 = dict.fromkeys(['rgb'])
        self.cam1_topic = "/camera1/color/image_raw"
        self.pub1_topic = "/siammask1/image"
        self.ROI1 = self.seg1 = self.points1 = None
        self.det_score1 = 0
        self.det_succ1 = 0

        self.instances1 = None


        # Camera 2 variables
        self.c2 = projection(camId=2)
        self.track2 = None
        self.fr2 = 0
        self.cam2_state = "wait"        
        self.cam2 = dict.fromkeys(['rgb'])       
        self.cam2_topic = "/camera2/color/image_raw"
        self.pub2_topic = "/siammask2/image"
        self.ROI2 = self.seg2 = self.points2 = None
        self.det_score2 = 0
        self.det_succ2 = 0

        self.instances2 = None


        # Other variables
        self.offset = None
        self.initial_location = None

        

        # Remove content from out/record directory
        if self.args.record:
            if os.path.exists('./data/record'):
                shutil.rmtree('./data/record')
            if not os.path.exists('./data/record'):
                os.makedirs('./data/record')

        # Get video file names
        if self.args.load_recorded_video:
            self.fname1 = self.args.video_1.split('/')[-1].split('.')[0]
            self.fname2 = self.args.video_2.split('/')[-1].split('.')[0]

            # Get directory for saving first frame and segmentation masks
            # bp()
            # self.savedir1 = '/'.join(self.args.video_1.split('/')[:-2])
            # self.savedir2 = '/'.join(self.args.video_2.split('/')[:-2])

            self.savedir = self.args.res_path
            if self.args.save_first_frame:
                if not os.path.exists(self.savedir+'/first_frame'):
                    os.makedirs(self.savedir+'/first_frame')
            if self.args.save_seg_mask:
                if not os.path.exists(self.savedir+'/seg_mask/'+self.fname1):
                    os.makedirs(self.savedir+'/seg_mask/'+self.fname1)
                if not os.path.exists(self.savedir+'/seg_mask/'+self.fname2):
                    os.makedirs(self.savedir+'/seg_mask/'+self.fname2)
            if self.args.save_volume_estimation:
                if not os.path.exists(self.savedir+'/volume_estimation'):
                    os.makedirs(self.savedir+'/volume_estimation')
            if self.args.export_pointcloud:
                if not os.path.exists(self.savedir+'/volume_estimation/'+self.fname1+'_pc'):
                    os.makedirs(self.savedir+'/volume_estimation/'+self.fname1+'_pc')


    # Load object detection model
    def LoadModel(self, args):
        if args.detpretrained == True:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        else:
            modelpath = args.detmodel # 'data/models/coco_maskrcnn_resnet50_fpn_2cat.pth'
            print(modelpath)

            model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)
            model.load_state_dict(torch.load(modelpath, map_location='cpu'))

        model.eval()
        model.cuda()

        return model


    def cam1_callback(self, data):
        if self.cam1_state == "ready" and self.cam2_state == "wait":
            return
        
        self.cam1['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

        if self.fr1 == 0:
            # Calibration
            if self.args.load_recorded_video:
                self.c1 = readCalibration(self.args.calib_1, self.c1)
            else:
                self.c1.cameraPose(self.cam1)

        # Detection
        if self.det_succ1 == 0:
            img = self.cam1['rgb']

            if self.args.save_first_frame and self.fr1 == 0:
                cv2.imwrite(self.savedir+'/first_frame/'+self.fname1+'.png', img)            

            try:
                self.det_succ1, self.instances1, self.img2plotc1 = self.detection.instanceSegmentationMaskRCNN(self.c1.camId, img, self.model, draw=True)
                self.ROI1, self.det_score1, self.seg1, self.points1 = self.detection.GetInstance(self.instances1, 0)
            except:
                print('Instance segmentation failed!')
            
            if self.ROI1 is not None:
                # Save segmentation mask
                if self.args.save_seg_mask:
                    cv2.imwrite(self.savedir+'/seg_mask/'+self.fname1+'/{:04d}.png'.format(self.fr1), self.seg1)

                # Initialise tracking
                rospy.loginfo('Initialising tracking from camera {:d}...'.format(self.c1.camId))
                self.track1 = maskTrack(self.ROI1)
                self.track1.run(img, 'init')

                self.cam1_state = "ready"
                rospy.loginfo('Camera {:d} ready!'.format(self.c1.camId))


    def cam2_callback(self, data):
        if self.cam2_state == "ready" and self.cam1_state == "wait":
            return
        
        self.cam2['rgb'] = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

        if self.fr2 == 0:
            # Calibration
            if self.args.load_recorded_video:
                self.c2 = readCalibration(self.args.calib_2, self.c2)
            else:
                self.c2.cameraPose(self.cam2)

        # Detection
        if self.det_succ2 == 0:
            img = self.cam2['rgb']

            if self.args.save_first_frame and self.fr2 == 0:
                cv2.imwrite(self.savedir+'/first_frame/'+self.fname2+'.png', img)

            try:
                self.det_succ2, self.instances2, self.img2plotc2 = self.detection.instanceSegmentationMaskRCNN(self.c2.camId, img, self.model, draw=True)
                self.ROI2, self.det_score2, self.seg2, self.points2 = self.detection.GetInstance(self.instances2, 0)
            except:
                print('Instance segmentation failed!')      
            
            if self.ROI2 is not None:
                # Save segmentation mask
                if self.args.save_seg_mask:
                    cv2.imwrite(self.savedir+'/seg_mask/'+self.fname2+'/{:04d}.png'.format(self.fr2), self.seg2)

                # Initialise tracking
                rospy.loginfo('Initialising tracking from camera {:d}...'.format(self.c2.camId))
                self.track2 = maskTrack(self.ROI2)
                self.track2.run(img, 'init')

                self.cam2_state = "ready"
                rospy.loginfo('Camera {:d} ready!'.format(self.c2.camId))


    def saveInitialLocation(self, centroid):
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
    

    def init_ros_topics(self):

        print("Retrieve frames")

        # ROS topic: retrieve frames
        rospy.Subscriber(self.cam1_topic, Image, self.cam1_callback)
        rospy.Subscriber(self.cam2_topic, Image, self.cam2_callback)

        # ROS topic: to publish image result
        self.pub1 = rospy.Publisher(self.pub1_topic, Image, queue_size=10)
        self.pub2 = rospy.Publisher(self.pub2_topic, Image, queue_size=10)


        # ROS topic: to publish 3D object location estimation and its dimensions
        self.markerPub = rospy.Publisher('estimatedObject', Marker, queue_size=10)
        self.state = Marker()
        self.state.type = Marker.CUBE
        self.state.header.frame_id = "/base_link"
        self.state.color.a = 1.
        self.state.color.r = 0.0;
        self.state.color.g = 1.0;
        self.state.color.b = 0.0;

        # Time for evaluation
        self.initial_locationPub = rospy.Publisher('initialLocation', Marker, queue_size=10)
        self.initial_state = Marker()
        self.initial_state.type = Marker.CUBE
        self.initial_state.header.frame_id = "/base_link"
        self.initial_state.color.a = 1.
        self.initial_state.color.r = 0.0;
        self.initial_state.color.g = 1.0;
        self.initial_state.color.b = 0.0;

    def readNextImage(self, vidcap1, vidcap2):
        success1, img1 = vidcap1.read()
        success2, img2 = vidcap2.read()
        img_msg1 = None
        img_msg2 = None

        if not success1:
            print('Finished reading image from video of camera 1.')
        if not success2:
            print('Finished reading image from video of camera 2.')
        
        if success1 and success2:
            img_msg1 = self.cvBridge.cv2_to_imgmsg(img1, encoding="bgr8")
            img_msg2 = self.cvBridge.cv2_to_imgmsg(img2, encoding="bgr8")
            
            self.cam1_callback(data=img_msg1)
            self.cam2_callback(data=img_msg2)

        return success1, success2, img_msg1, img_msg2

    # Live run
    def runLive(self):
        # Initialise subscribing topics
        self.init_ros_topics()

        # Read file to calibrate camera to robot
        f = open('./data/calibration/cameras_robot.pckl', 'rb')
        self.camera_robot_transformation = pickle.load(f)
        f.close()

        # Wait for cameras to be ready before going ahead
        while ((self.cam1_state != "ready") or (self.cam2_state != "ready")):
            continue

        # Estimate object dimensions [only on the first frame]
        rospy.loginfo('Estimating object dimensions...')
        self.getObjectDimensions()

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():

            # Track in 2D
            img1 = self.track1.run(self.cam1['rgb'], 'track')
            img2 = self.track2.run(self.cam2['rgb'], 'track')

            # Triangulate to get 3D centroid
            self.container, img1, img2 = get3D(self.c1, self.c2, self.track1.mask, self.track2.mask, self.container, img1, img2, drawCentroid=True, drawDimensions=False)

            # ROS publish results
            self.pub1.publish(self.cvBridge.cv2_to_imgmsg(img1, encoding="passthrough"))
            self.pub2.publish(self.cvBridge.cv2_to_imgmsg(img2, encoding="passthrough"))

            ########### 
            # PUBLISH #
            ###########
            # Change refence system from cameras to robot
            self.container.centroid = np.matmul(self.camera_robot_transformation, np.append(self.container.centroid, 1.).reshape(4,1))

            self.initial_locationPub.publish(self.initial_state)

            # 3D marker message
            self.state.pose.position.x = self.container.centroid[0,0]
            self.state.pose.position.y = self.container.centroid[1,0]
            self.state.pose.position.z = self.container.centroid[2,0]
            self.state.scale.x = self.container.maxwidth
            self.state.scale.y = self.container.maxwidth
            self.state.scale.z = self.container.h
            self.markerPub.publish(self.state)

            if self.args.record:
                cv2.imwrite(self.package_path+'/data/record/c1_track_{}.png'.format(self.fr1), img1)
                cv2.imwrite(self.package_path+'/data/record/c2_track_{}.png'.format(self.fr2), img2)
                
            rate.sleep()

        rospy.spin() # Keep python from exiting until this node is stopped

    # Outputs:
        #  1: success initialisation, saved container centroid
        # -1: unsuccess initialisation of the container
        # -2: failed to read images
    def initialisation(self, vidcap1, vidcap2):

        lode_succ = 0
        self.det_succ1 == 0
        self.det_succ2 == 0

        while self.det_succ1 == 0 or self.det_succ2 == 0 or lode_succ < 1:
            print("Frame #" + str(self.fr))
            success1, success2, img_msg1, img_msg2 = self.readNextImage(vidcap1, vidcap2)

            if success1 == 0 or success2 == 0:
                self.results_file.close()
                self.results_file.close()
                self.results_file_width.close()
                return -2

            lode_succ = self.container.LoDE(self.cam1['rgb'], self.cam2['rgb'], self.c1, self.c2, self.instances1, self.instances2, self.args.LoDE_hstep, self.args.LoDE_rstep)
            
            if lode_succ == 1:
                filling_lev_c1, filling_type_c1, bbox1 = self.container.estimateFillingPropertiesPerCam(self.cam1['rgb'], self.seg1, 1, self.args)
                filling_lev_c2, filling_type_c2, bbox2 = self.container.estimateFillingPropertiesPerCam(self.cam2['rgb'], self.seg2, 2, self.args)              

                self.fout_res_lev.write('{:03d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, '.format(self.fr, bbox1[0], bbox1[1], bbox1[2], bbox1[3], filling_lev_c1, filling_type_c1))
                self.fout_res_lev.write('{:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(bbox2[0], bbox2[1], bbox2[2], bbox2[3], filling_lev_c2, filling_type_c2))

                self.container.estimateFillingProperties(self.fr)
                self.container.saveResults(self.results_file, self.fr)
                self.container.saveWidth(self.results_file_width, self.fr)

                if self.args.export_pointcloud:
                    if self.args.pointcloud_frame == [] or (self.fr in self.args.pointcloud_frame):
                        print('Saving pointcloud at frame '+str(self.fr))
                        self.pointcloud_path = self.savedir+'/volume_estimation/'+self.fname1+'_pc/'+self.fname1+'_pc_f'+str(self.fr)+'.csv'
                        self.container.exportShapePointCloud(self.pointcloud_path)

                self.saveInitialLocation(self.container.centroid)

                return 1
            else:
                self.fout_res_lev.write('{:03d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, '.format(self.fr, -1, -1, -1, -1, -1, -1))
                self.fout_res_lev.write('{:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(-1, -1, -1, -1, -1, -1))
                
                self.container.saveResults(self.results_file, self.fr)
                self.container.saveWidth(self.results_file_width, self.fr)
                
                return -1

    def reInitialisation(self):
        print('Re-initialisation!')

        img1 = self.cam1['rgb']
        img2 = self.cam2['rgb']

        self.instances1 = None
        self.instances2 = None

        try:
            self.det_succ1, self.instances1, self.img2plotc1 = self.detection.instanceSegmentationMaskRCNN(self.c1.camId, img1, self.model, draw=True)
            self.ROI1, self.det_score1, self.seg1, self.points1 = self.detection.GetInstance(self.instances1, 0)

            self.det_succ2, self.instances2, self.img2plotc2 = self.detection.instanceSegmentationMaskRCNN(self.c2.camId, img2, self.model, draw=True)
            self.ROI2, self.det_score2, self.seg2, self.points2 = self.detection.GetInstance(self.instances2, 0)

            if self.det_succ1 == 0 or self.det_succ2 == 0:
                return False

            lode_succ = self.container.LoDE(self.cam1['rgb'], self.cam2['rgb'], self.c1, self.c2, self.instances1, self.instances2, self.args.LoDE_hstep, self.args.LoDE_rstep)

            if lode_succ == 1:
                filling_lev_c1, filling_type_c1, bbox1 = self.container.estimateFillingPropertiesPerCam(img1, self.seg1, 1, self.args)
                filling_lev_c2, filling_type_c2, bbox2 = self.container.estimateFillingPropertiesPerCam(img2, self.seg2, 2, self.args)              
                
                self.fout_res_lev.write('{:03d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, '.format(self.fr, bbox1[0], bbox1[1], bbox1[2], bbox1[3], filling_lev_c1, filling_type_c1))
                self.fout_res_lev.write('{:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(bbox2[0], bbox2[1], bbox2[2], bbox2[3], filling_lev_c2, filling_type_c2))

                self.container.estimateFillingProperties(self.fr)

                self.container.saveResults(self.results_file, self.fr)
                self.container.saveWidth(self.results_file_width, self.fr)

                if self.args.export_pointcloud:
                    if self.args.pointcloud_frame == [] or (self.fr in self.args.pointcloud_frame):
                        print('Saving pointcloud at frame '+str(self.fr))
                        self.pointcloud_path = self.savedir+'/volume_estimation/'+self.fname1+'_pc/'+self.fname1+'_pc_f'+str(self.fr)+'.csv'
                        self.container.exportShapePointCloud(self.pointcloud_path)


                self.track1 = maskTrack(self.ROI1)
                self.track1.run(img1, 'init')

                self.track2 = maskTrack(self.ROI2)
                self.track2.run(img2, 'init')

                return True
            
            else:
                return False

        except:
            print('Instance segmentation failed!')
            return False 


    def tracking(self, vidcap1, vidcap2):
        success1, success2, img_msg1, img_msg2 = self.readNextImage(vidcap1, vidcap2)

        if success1 == 0 or success2 == 0:
            self.results_file.close()
            self.results_file.close()
            self.results_file_width.close()
            return -2

        # Track in 2D
        img1 = self.track1.run(self.cam1['rgb'], 'track')
        img2 = self.track2.run(self.cam2['rgb'], 'track')

        self.instances1 = {'masks' : [self.track1.mask], 'boxes' : [self.track1.ROI],  'labels' : [], 'scores' : [], 'centroid' : []}
        self.instances2 = {'masks' : [self.track2.mask], 'boxes' : [self.track2.ROI],  'labels' : [], 'scores' : [], 'centroid' : []}
        # objmask1 = self.track1.mask
        # objmask2 = self.track2.mask

        # Save segmentation mask
        if self.args.save_seg_mask:
            cv2.imwrite(self.savedir+'/seg_mask/'+self.fname1+'/{:04d}.png'.format(self.fr1), self.track1.mask * 255)
            cv2.imwrite(self.savedir+'/seg_mask/'+self.fname2+'/{:04d}.png'.format(self.fr2), self.track2.mask * 255)

        lode_succ = self.container.LoDE(self.cam1['rgb'], self.cam2['rgb'], self.c1, self.c2, self.instances1, self.instances2, self.args.LoDE_hstep, self.args.LoDE_rstep)
    
        if lode_succ == 1:
            filling_lev_c1, filling_type_c1, bbox1 = self.container.estimateFillingPropertiesPerCam(img1, self.track1.mask, 1, self.args)
            filling_lev_c2, filling_type_c2, bbox2 = self.container.estimateFillingPropertiesPerCam(img2, self.track2.mask, 2, self.args)              
            
            self.fout_res_lev.write('{:03d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, '.format(self.fr, bbox1[0], bbox1[1], bbox1[2], bbox1[3], filling_lev_c1, filling_type_c1))
            self.fout_res_lev.write('{:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(bbox2[0], bbox2[1], bbox2[2], bbox2[3], filling_lev_c2, filling_type_c2))

            self.container.estimateFillingProperties(self.fr)

            self.container.saveResults(self.results_file, self.fr)
            self.container.saveWidth(self.results_file_width, self.fr)

            if self.args.export_pointcloud:
                if self.args.pointcloud_frame == [] or (self.fr in self.args.pointcloud_frame):
                    print('Saving pointcloud at frame '+str(self.fr))
                    self.pointcloud_path = self.savedir+'/volume_estimation/'+self.fname1+'_pc/'+self.fname1+'_pc_f'+str(self.fr)+'.csv'
                    self.container.exportShapePointCloud(self.pointcloud_path)

            return 1

        elif lode_succ == -2:
            if self.reInitialisation() == False: 
                self.fout_res_lev.write('{:03d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, '.format(self.fr, -1, -1, -1, -1, -1, -1))
                self.fout_res_lev.write('{:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(-1, -1, -1, -1, -1, -1))

                return -1
            else:
                return 1
        
        

    def runRecorded(self):
        # log results
        self.results_file = open(self.savedir+'/volume_estimation/'+self.fname1+'_properties.txt', 'w')
        self.results_file.write('# Frame | x [m] | y [m] | z [m] | Width bottom [mm] | Width middle [mm] | Width top [mm] | Max width [mm] | Height [mm] | Volume [mL] | Filling level | Filling type | Filling Mass | Weight\n')
        
        self.fout_res_lev = open(self.savedir+'/volume_estimation/'+self.fname1 + '_fillinglevel.txt', 'w')
        self.fout_res_lev.write('# Frame | Bbox1 [x1, y1, x2, y2] | Filling Level c1 | | Filling type c1 | Bbox2 [x1, y1, x2, y2] | Filling Level c2 | Filling type c2\n')
        
        self.results_file_width = open(self.savedir+'/volume_estimation/'+self.fname1+'_width.txt', 'w')
        self.results_file_width.write('# Frame | z [m] | Radius [m]\n')


        try:
            vidcap1 = cv2.VideoCapture(self.args.video_1)
            vidcap2 = cv2.VideoCapture(self.args.video_2)

            num_frames = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if num_frames != int(vidcap2.get(cv2.CAP_PROP_FRAME_COUNT)):
                print("Different number of frames")              

            containerInit = -1
            trackingSucc = 1

            while containerInit == -1:
                print("Frame #" + str(self.fr) + "/" + str(num_frames))
                containerInit = self.initialisation(vidcap1, vidcap2)

                if containerInit == -2:
                    return

                self.fr  += 1
                self.fr1 += 1
                self.fr2 += 1
           
            while trackingSucc != -2:
                print("Frame #" + str(self.fr) + "/" + str(num_frames))

                start_time = time.time()

                if trackingSucc == 1:
                    trackingSucc = self.tracking(vidcap1, vidcap2)
                elif trackingSucc == -1:
                    containerInit = self.initialisation(vidcap1, vidcap2)

                    if containerInit == -2:
                        return

                    trackingSucc = containerInit

                end_time = time.time()
                print("Frame processing (milliseconds):", (end_time-start_time)*1000)

                self.fr += 1
                self.fr1 += 1
                self.fr2 += 1


            self.results_file.close()
            self.results_file_width.close()
            self.fout_res_lev.close()
        
        except Exception as e:
            self.results_file.write(traceback.format_exc())
            self.results_file_width.close()
            self.fout_res_lev.close()

    # Load recorded videos
    def run(self):

        if not(self.args.load_recorded_video):
            self.runLive()
        else:
            self.runRecorded()
            
