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
import numpy as np
import copy
import cv2

import torch
import torchvision
import torchvision.transforms as T

from pdb import set_trace as bp

trf = T.Compose([T.ToPILImage(),
                 T.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'wine_glass', 'cup']


# 
# Make it square [we need this by now, we will remove it later when possible]
# def prepareImage(_img):
#     img = copy.deepcopy(_img)

#     center_y = int(img.shape[1]/2)
#     h_half = int(img.shape[0]/2)
    
#     return  img[:,center_y-h_half:center_y+h_half], center_y-h_half

def ComputeBboxFromMask(_objmask):
    objmask = copy.deepcopy(_objmask)

    points = getExtremePoints(objmask).reshape((4,2)) # [most up, most right, most down, most left]
    pmin = np.amin(points, axis=0)
    pmax = np.amax(points, axis=0)

    bbox = np.concatenate( (pmin, pmax), axis=0)

    return bbox

def getExtremePoints(seg):
    seg_t = seg.copy()
  
    # contours, _ = cv2.findContours(seg_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(seg_t, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0]
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contours = max(contour_sizes, key=lambda x: x[0])[1]

    #The most right point
    point_r = (contours[contours[:,:,0].argmin()][0]).reshape(1,2)
    #The most left point
    point_l = (contours[contours[:,:,0].argmax()][0]).reshape(1,2)
    #The most up point
    point_u = (contours[contours[:,:,1].argmin()][0]).reshape(1,2)
    #The most down point
    point_d = (contours[contours[:,:,1].argmax()][0]).reshape(1,2)
    
    return np.concatenate((point_u, point_r, point_d, point_l), axis=0)


class Camera:
    def __init__(self, _detpretrained):
        self.confidence_th = 0.5 # To filter out instances with weak confidence

        self.detpretrained = _detpretrained

    ###
    def instanceSegmentationMaskRCNN(self, camId, _img, model, draw=False):
        print('instanceSegmentationMaskRCNN') 

        img = copy.deepcopy(_img)
        # img = img[:, :, [2, 1, 0]]

        print('Detecting object from camera {:d}...'.format(camId))
        # img_square, self.offset = prepareImage(self.cam1['rgb'])

        output = model([trf(img).to(device)])[0] # outputs are already sorted by confidence score

        print('Number of instances: {:d}'.format(len(output['labels'])))
        
        if len(output['labels']) == 0:
            return False, None, -1, None, None, img

        valid_strong_instances = {'masks' : [], 'boxes' : [],  'labels' : [], 'scores' : [], 'centroid' : []}

        if self.detpretrained == True:
            # valid_labels_idx = [44, 46, 47, 86] # bottle, wine glass, cup, vase
            valid_labels_idx = [46, 47] # bottle, wine glass, cup, vase
        else:
            valid_labels_idx = [1,2] # wine glass, cup

        num_cups_th = 0
        num_glasses_th = 0
        success = False

        # Draw glasses and similar (e.g. vase, cup)
        for i in range(0,len(output['labels'])):
            if output['scores'][i] >= self.confidence_th and output['labels'][i] in valid_labels_idx:  
                if output['labels'][i] == 1 or output['labels'][i] == 46:
                    print('Object found! (glass, {:.2f})'.format(output['scores'][i]))

                    if output['scores'][i] >= self.confidence_th :
                        success = True
                        num_glasses_th += 1

                elif output['labels'][i] == 2 or output['labels'][i] == 47:
                    print('Object found! (cup, {:.2f})'.format(output['scores'][i]))

                    if output['scores'][i] >= self.confidence_th :
                        success = True
                        num_cups_th += 1

                valid_strong_instances['masks'].append(output['masks'][i,:,:,:])
                valid_strong_instances['boxes'].append(output['boxes'][i])
                valid_strong_instances['labels'].append(output['labels'][i])
                valid_strong_instances['scores'].append(output['scores'][i])
                
                if draw:
                    img = self.DrawInstanceMaskBBox(img, output['masks'][i,:,:,:], output['boxes'][i], 
                        output['labels'][i], output['scores'][i])
        
        print('\n')
        print('Number of glasses above threshold ({:f}): {:d}'.format(self.confidence_th,num_glasses_th))
        print('Number of cups above threshold ({:f}): {:d}'.format(self.confidence_th,num_cups_th))

        # ofile_c1 = open('num_dets.txt', 'a')
        # if camId == 1:
        #     ofile_c1.write('{:d} {:d} '.format(num_glasses_th, num_cups_th))
        # else:
        #     ofile_c1.write('{:d} {:d}\n'.format(num_glasses_th, num_cups_th))
        # ofile_c1.close()

        if not success:
            return False, None, img
        else:
            return True, valid_strong_instances, img

    # 
    def DrawInstanceMaskBBox(self, _img, mask, bbox, label, score):
        img = copy.deepcopy(_img)

        seg = np.uint8(255.* (mask >= 0.5).detach().cpu().numpy().squeeze())

        img[:, :, 1] = (seg > 0) * 255 + (seg == 0) * img[:, :, 1]*0.65
        img[:, :, 0] = img[:, :, 0]*0.65
        img[:, :, 2] = img[:, :, 2]*0.65

        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,0,0),3)

        if label == 1 or label == 46:
            cv2.putText(img, 'Glass', (bbox[2], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        elif label == 2 or label == 47:
            cv2.putText(img, 'Cup', (bbox[2], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # cv2.putText(img, COCO_INSTANCE_CATEGORY_NAMES[label], (bbox[2], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA) 
        cv2.putText(img, str('{:.2f}'.format(score)), (bbox[2], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # for j, point in enumerate(points):
            #     cv2.circle(img, tuple(point), 10, (0,0,255), -1)
            #     cv2.putText(img, str(j), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return img

    def GetInstance(self, instances, j):
        ROI = []
        # seg = np.uint8(255.*(instances['masks'][j,:,:,:] >= 0.5).detach().cpu().numpy().squeeze())
        seg = np.uint8(255.*(instances['masks'][j] >= 0.5).detach().cpu().numpy().squeeze())
        points = getExtremePoints(seg)
        best_score = float(instances['scores'][j].detach().cpu().numpy())
        ROI.append(instances['boxes'][j].cpu().detach().numpy())

        return np.array(ROI).astype(int), best_score, seg, points

