# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

class siamMask:
	def __init__(self, ROI):
		parser = argparse.ArgumentParser()
		args, unknown = parser.parse_known_args()
		#args.resume = 'libs/siamMask/experiments/siammask/SiamMask_DAVIS.pth'
		#args.config = 'libs/siamMask/experiments/siammask/config_davis.json'

		args.resume = 'libs/perception/siamMask/experiments/siammask/SiamMask_VOT_LD.pth'
		args.config = 'libs/perception/siamMask/experiments/siammask/config_vot18.json'

		# Setup device
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		torch.backends.cudnn.benchmark = True

		# Setup Model
		self.cfg = load_config(args)
		from experiments.siammask.custom import Custom
		siammask = Custom(anchors=self.cfg['anchors'])
		if args.resume:
			assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
			siammask = load_pretrain(siammask, args.resume)

		# Extend ROI 20%
		w = ROI[0,2]-ROI[0,0]
		h = ROI[0,3]-ROI[0,1]
		margin_x = int(0.1*w) #Enlarge 20%
		margin_y = int(0.1*h) #Enlarge 20%


		self.ROI = [ROI[0,0]-margin_x, ROI[0,1]-margin_y, w+2*margin_x, h+2*margin_y]
		self.siammask = siammask.eval().to(device)
		self.state = None
		self.mask = None

	def run(self, img, mode):

		if mode is 'init':  # init
			x, y, w, h = self.ROI
			target_pos = np.array([x + w / 2, y + h / 2])
			target_sz = np.array([w, h])
			img_crop = img[int(y):int(y+h),int(x):int(x+w),:]
			self.state = siamese_init(img, target_pos, target_sz, self.siammask, self.cfg['hp'])  # init tracker

		elif mode is 'track':  # tracking
			self.state = siamese_track(self.state, img, mask_enable=True, refine_enable=True)  # track
			#location = self.state['ploygon'].flatten()
			self.mask = (self.state['mask'] > self.state['p'].seg_thr).astype(np.uint8)

			img[:, :, 1] = (self.mask > 0) * 255 + (self.mask == 0) * img[:, :, 1]
			return img



