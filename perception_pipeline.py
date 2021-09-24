#!/usr/bin/env python
#
################################################################################## 
#        Author: Ricardo Sanchez Matilla
#        Author: Yik Lung Pang
#        Author: Alessio Xompero
#         Email: corsmal-benchmark@qmul.ac.uk
#
#
#  Created Date: 2020/02/13
# Modified Date: 2020/10/05
#
# Centre for Intelligent Sensing, Queen Mary University of London, UK
# 
################################################################################## 
# License
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
# International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to 
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##################################################################################
#
# System libs
import glob
import sys
import argparse

import pandas as pd
import csv


# Numeric libs
import cv2
import torch
import torchvision

# Computer Vision libs
from libs.perception.tracker import *

#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fnames_filling =  ['fi0_fu0','fi1_fu1', 'fi1_fu2','fi2_fu1', 'fi2_fu2', 'fi3_fu1', 'fi3_fu2']

def populate_filenames(mode):
    list_filenames = []
    for s in range(0,2):
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

#### TRAINING DATA ###

def TrainingDataParser(args):
    for objid in range(1,7):
        # containerpath = args.datapath + '/{:d}'.format(objid)

        args.res_path = args.datapath + '/vision_estimations/container' + '/{:d}'.format(objid)
        
        if objid < 7:
            list_files = populate_filenames(0)
        else:
            list_files = populate_filenames(1)

        for f in list_files:
            args.video_1 = args.datapath + '/videos/' + str(objid) + '/rgb/'+ f + '_c1.mp4'
            args.video_2 = args.datapath + '/videos/' + str(objid) + '/rgb/'+ f + '_c2.mp4'
            args.calib_1 = args.datapath + '/calibration/' + str(objid) + '/extra/'+ f + '_c1_calib.pickle'
            args.calib_2 = args.datapath + '/calibration/' + str(objid) + '/extra/'+ f + '_c2_calib.pickle'
            
            track = Tracker(args)
            track.run()


#### PUBLIC TESTING SET ###
def PublicTestingDataParser(args):
    for objid in range(10,12):
        # containerpath = args.datapath + '/{:d}'.format(objid)

        args.res_path = args.datapath + '/vision_estimations/container' + '/{:d}'.format(objid)
        
        listimages = pd.read_csv('data/annotations/annotation_public_test.csv', sep=',')
        boolArray = listimages['Container ID'].values == objid

        list_files = []
        for j in listimages['Sequence'].values[boolArray]:
            list_files.append('{:04d}'.format(j)) 

        for f in list_files:
            args.video_1 = args.datapath + '/videos/' + str(objid) + '/rgb/'+ f + '_c1.mp4'
            args.video_2 = args.datapath + '/videos/' + str(objid) + '/rgb/'+ f + '_c2.mp4'
            args.calib_1 = args.datapath + '/calibration/' + str(objid) + '/extra/'+ f + '_c1_calib.pickle'
            args.calib_2 = args.datapath + '/calibration/' + str(objid) + '/extra/'+ f + '_c2_calib.pickle'
            
            track = Tracker(args)
            track.run()

if __name__ == '__main__':
    print('Initialising:')
    print('Python {}.{}'.format(sys.version_info[0], sys.version_info[1]))
    print('OpenCV {}'.format(cv2.__version__))
    print('PyTorch {}'.format(torch.__version__))
    print('Torchvision {}'.format(torchvision.__version__))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_recorded_video', type=int, choices=[0,1], default=0)
    parser.add_argument('--save_first_frame', type=int, choices=[0,1], default=0)
    parser.add_argument('--save_seg_mask', type=int, choices=[0,1], default=0)
    parser.add_argument('--save_volume_estimation', type=int, choices=[0,1], default=0)
    parser.add_argument('--export_pointcloud', type=int, choices=[0,1], default=0)
    parser.add_argument('--pointcloud_frame', nargs='+', type=int, default=[])
    parser.add_argument('--use_LoDE', type=int, choices=[0,1], default=0)
    parser.add_argument('--LoDE_hstep', type=float, default=0.001)
    parser.add_argument('--LoDE_rstep', type=float, default=18.0)
    parser.add_argument('--record', type=int, choices=[0,1], default=0)
    parser.add_argument('--max_num_frames', type=int, default=-1)
    # Path to videos
    parser.add_argument('--video_1', type=str)
    parser.add_argument('--video_2', type=str)
    parser.add_argument('--calib_1', type=str)
    parser.add_argument('--calib_2', type=str)
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--dataset', default='train', type=str, help='Dataset to process: train, test or all')
    # FillingNet
    parser.add_argument('--network_input_dimensions', default=128, type=int, help='network input dimensions')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained weights')
    parser.add_argument('--useMask', default=False, action='store_true')
    parser.add_argument('--level_model_path', type=str, default='data/models/resnet18scratch_CE_model.t7', help='path to a trained model')
    parser.add_argument('--type_model_path', type=str, default='data/models/resnet18_filling.t7', help='path to a trained model')
    parser.add_argument('--typelevel_model_path', type=str, default='data/models/filling_type_level_model.t7', help='path to filling level and type model')
    parser.add_argument('--fillingmode', type=str, default='joint', choices=['joint','independent'], help='path to filling level and type model')

    
    # Detection model
    parser.add_argument('--detmodel', type=str, default='data/models/coco_maskrcnn_resnet50_fpn_2cat.pth', help='path to Mask R-CNN trained model')
    parser.add_argument('--detpretrained', default=True, action='store_true')

    args = parser.parse_args()

    if device == 'cuda':
        torch.cuda.set_device(0)
    print('Using {}'.format(device))

    if not[x for x in (args.video_1, args.video_2, args.calib_1, args.calib_2) if x is None]:
        track = Tracker(args)
        track.run()
    else:
        if args.dataset == 'train':
            TrainingDataParser(args)
        elif args.dataset == 'test':
            PublicTestingDataParser(args)
        else:
            TrainingDataParser(args)
            PublicTestingDataParser(args)
