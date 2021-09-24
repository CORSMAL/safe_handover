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

# System libraries
from __future__ import print_function
import os
import copy
import argparse
from tqdm import tqdm

# Numerical libraries
import numpy as np
import cv2

# PyTorch library
import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models


from pdb import set_trace as bp

# Own libraries
# import importlib


device = 'cuda' if torch.cuda.is_available() else 'cpu'
DISABLE_TQDM = False

JOINT2FILLINGLEVELIDX = [0, 1, 2, 1, 2, 1, 2, 3]
JOINT2FILLINGTYPEIDX = [0, 1, 1, 2, 2, 3, 3, 4]

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


###########################
##    Transformations    ##
###########################
def setTransformations(net_in_dims):
	transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((net_in_dims, net_in_dims)),
			transforms.ToTensor(),
		])

	return transform


def getTrainingStatistics(filepathname):
	try:
		normalisationValues = np.loadtxt(filepathname)

		trainingStatistics = dict.fromkeys(['mean', 'std'])
		trainingStatistics['mean'] = normalisationValues[0, 0:3]
		trainingStatistics['std']  = normalisationValues[1, 0:3]

		return trainingStatistics

	except:
		print('Training statistics for data normalisation file not found')
		assert 1 == 0

		return None

def crop(frame, bbox, margin):
	
	ROI = frame[int(bbox[1]*(1.-margin)):int(bbox[3]*(1.+margin)), int(bbox[0]*(1.-margin)):int(bbox[2]*(1.+margin))]
	
	return ROI

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class resnet18(nn.Module):
	def __init__(self, args, num_classes):

		super(resnet18, self).__init__()
		self.args = args

		self.numClasses = num_classes

		self.backbone = models.resnet18(pretrained=True)
		self.backbone.fc = Identity()

		# Freeze all parameters
		if self.args.pretrained:
			for param in self.backbone.parameters():
				param.requires_grad = False

		self.classifier = nn.Linear(in_features=512, out_features=self.numClasses)


	def forward(self, x):
		feat = self.backbone(x)
		pred = self.classifier(feat)
		pred = F.softmax(pred, dim = 1)

		return pred, feat


class TransformLayer(torch.nn.Module):

	def __init__(self, mean, std):
		super().__init__()
		
		self.mean = torch.nn.Parameter(mean, requires_grad=False)
		self.std = torch.nn.Parameter(std, requires_grad=False)

	def forward(self, x):
		return x.sub(self.mean).div(self.std)



class FillingNet:
	def __init__(self, args):
		self.args = args

		self.margin=0.05	

		# Instatiate model
		if args.fillingmode == 'independent':
			self.levelnet = resnet18(self.args, 4)
			self.levelnet.load_state_dict(torch.load(self.args.level_model_path, map_location='cpu')['net'])
			self.levelnet = self.levelnet.to(device)
			self.levelnet.eval()

			self.typenet = resnet18(self.args, 5)
			self.typenet.load_state_dict(torch.load(self.args.type_model_path, map_location='cpu')['net'])
			self.typenet = self.typenet.to(device)
			self.typenet.eval()
		elif args.fillingmode == 'joint':
			self.typelevelnet = resnet18(self.args, 8)
			self.typelevelnet.load_state_dict(torch.load(self.args.typelevel_model_path, map_location='cpu')['net'])
			self.typelevelnet = self.typelevelnet.to(device)
			self.typelevelnet.eval()

		self.fillingLevel = -1
		self.fillingType = -1
		
		self.flevpreds = None
		self.ftyppreds = None

		self.jointFilling = -1
		self.jointpreds = None
	

	def getFillingLevel(self):
		return self.fillingLevel

	def getFillingType(self):
		return self.fillingType

	def getJointFillingTypeLevel(self):
		return self.jointFilling

	def getFillingLevelPredictions(self):
		return self.flevpreds

	def getFillingTypePredictions(self):
		return self.ftyppreds

	def getJointFillingTypeLevelPredictions(self):
		return self.jointpreds

	def run(self, img, objmask):

		trainingStatistics = getTrainingStatistics('data/models/trainingStatistics.txt')

		trans = TransformLayer(	mean=torch.as_tensor(trainingStatistics['mean'], dtype=torch.float, device=device)[None, :, None, None],
														std =torch.as_tensor(trainingStatistics['std'],  dtype=torch.float, device=device)[None, :, None, None]
													)


		transform = setTransformations(self.args.network_input_dimensions)

		rgb = copy.deepcopy(img)
		if transform:
			rgb = transform(crop(rgb, objmask, self.margin))

		img_crop = rgb.to(device)

		
		if self.args.fillingmode == 'independent':
			outputs, _ = self.levelnet(trans(img_crop))
			_, pred_levels = outputs.max(1)

			outputs2, _ = self.typenet(trans(img_crop))
			_, pred_types = outputs2.max(1)

			self.fillingLevel = pred_levels.cpu().numpy()[0]
			self.fillingType = pred_types.cpu().numpy()[0]

			self.flevpreds = outputs.cpu().detach().numpy()
			self.ftyppreds = outputs2.cpu().detach().numpy()

			J = np.reshape(np.outer(self.flevpreds,self.ftyppreds).transpose(), (1,20),'C')
			W = GetFeasibleStatusMapping()

			self.jointpreds = np.matmul(J, W).transpose()
			self.jointFilling = np.argmax(self.jointpreds)

		elif self.args.fillingmode == 'joint':
			print('Joint estimation!')
			outputs, _ = self.typelevelnet(trans(img_crop))
			_, preds = outputs.max(1)

			self.jointFilling = preds.cpu().numpy()[0]
			self.jointpreds = outputs.cpu().detach().numpy()

			self.fillingLevel = JOINT2FILLINGLEVELIDX[self.jointFilling]
			self.fillingType = JOINT2FILLINGTYPEIDX[self.jointFilling]
