# Towards safe human-to-robot handovers of unknown containers

This is a real-to-simulation framework to conduct safe human-to-robot handovers with vision predictions of the physical properties of unknown cups or drinking glasses and of the human hands from videos of a person manipulating the object. We complete the handover in simulation and we quantify the safeness of the human and object using annotations of the real object properties. To further increase the safety for the human, we estimate a safe grasp region, i.e. not occluded by the hand of the person holding the object, for the robot to grasp the object during the handover.

![overview](overview.gif)

[[arxiv](https://arxiv.org/abs/2107.01309)] [[webpage](http://corsmal.eecs.qmul.ac.uk/safe_handover.html)][[pre-trained models]()][[CCM dataset](https://corsmal.eecs.qmul.ac.uk/containers_manip.html)]

## Installation

### Requirements
* Ubuntu 18.04
* Anaconda 3
* Python 3.6.10
* PyTorch 1.3.1
* PyBullet 2.4.8
* OpenCV 4.1.0



## Setup
Clone the repository then install the required packages into your anaconda environment and activate it.
```
conda create --name real2sim --file conda_env_setup.txt
conda activate real2sim
```
Set up SiamMask library
```
cd libs/perception/siamMask
./make.sh
export SiamMask=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../../../../../
```
Download pre-trained models for object segmentation
```
wget http://corsmal.eecs.qmul.ac.uk/resources/models/coco_maskrcnn_resnet50_fpn_2cat.zip -P data/models/
unzip data/models/coco_maskrcnn_resnet50_fpn_2cat.zip -d data/models/
rm data/models/coco_maskrcnn_resnet50_fpn_2cat.zip
```
Download hand keypoint estimation results and pre-trained models for filling type and level estimation
```
wget https://zenodo.org/record/5525333/files/supplemetary_data.zip?download=1 -O data/supplemetary_data.zip
unzip data/supplemetary_data.zip -d data/
mv data/supplemetary_data/models/* data/models/
mv data/supplemetary_data/openpose/* data/vision_estimations/openpose/
rm data/supplemetary_data.zip
rm data/supplemetary_data -r
```
Download the annotation file for the training dataset
```
wget http://corsmal.eecs.qmul.ac.uk/data/CCM/ccm_train_annotation.json -P data/annotations/
wget http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/Annotation_Training.csv -P data/annotations/
```
We provide a demo video from the CORSMAL Container Manipulation (CCM) dataset for testing within the repository. However, if you would like to run experiments on other videos, you can run the following scripts to download the full dataset. (Note: We do not provide [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) as part of the vision pipeline within this repository. Instead, we only provide the estimated hand keypoints results for the public training and testing recordings in scenario 1 and 2. If you would like to experiment with other videos, please setup OpenPose or other hand keypoints estimation method along side this repository.)

Download the recording and calibration files from the CORSMAL Containers Manipulation (CCM) dataset
```
./download_CCM.sh
```
Download the file for mapping filenames between previous and updated version of the CCM dataset
```
wget http://corsmal.eecs.qmul.ac.uk/data/CCM/ccm_annotations_train_set_mapping.txt -P data/CCM/
```


## Demo
This is a demo for estimating properties and simulating the handover for a single recording

First, run the vision based perception pipeline to estimate the properties of handover (trajectory and shape of the container, hand keypoints, filling type and level)
```
python perception_pipeline.py --load_recorded_video 1 --save_seg_mask 0 --save_volume_estimation 1 --export_pointcloud 1 --pointcloud_frame 0 --use_LoDE 1 --LoDE_hstep 0.002 --LoDE_rstep 5.0 --video_1 data/CCM/train/view1/rgb/000618.mp4 --video_2 data/CCM/train/view2/rgb/000618.mp4 --calib_1 data/CCM/train/view1/calib/000618.pickle --calib_2 data/CCM/train/view2/calib/000618.pickle --res_path data/vision_estimations/
```
Then, run the simulation to evaluate the handover. A GUI of the simulation will show up and the recording of the handover from camera view 1 will be played on the side.
```
python evaluate_handover.py --render --show_recording --run_desc demo --recording_id 618
```
Numerical results are saved in the `results/` folder.

## Vision based perception pipeline
The vision based perception pipeline takes a pair of videos and calibration files as input. (Otherwise use the `--dataset` flag to indicate whether to process `train` or `test` or `all` recordings in the CCM dataset)

Acceptable flags:
- load_recorded_video - process recorded videos
- save_first_frame - save the first frame of the video as png <video_path>/first_frame/
- save_seg_mask - save the first frame of the video with segmentation <video_path>/seg_mask/
- save_volume_estimation - save the dimension estimation and centroid tracking coordinates in a log file in <video_path>/volume_estimation/
- use_LoDE - use LoDE for dimension and volume estimation
- LoDE_hstep - height step size of LoDE
- LoDE_rstep - angular step size of LoDE
- export_pointcloud - export the pointcloud of the object as estimated from LoDE
- pointcloud_frame - which frame to export the pointcloud from
- video_1 - path to video from camera 1
- video_2 - path to video from camera 2
- calib_1 - path to calibration file for camera 1
- calib_2 - path to calibration file for camera 2
- res_path - path to save the results
- datapath - path that contains the videos and calibration files folders
- dataset - Dataset to evaluate: train, test or all (if not processing a single recording)


Example:
```
python perception_pipeline.py --load_recorded_video 1 --save_seg_mask 0 --save_volume_estimation 1 --export_pointcloud 1 --pointcloud_frame 0 --use_LoDE 1 --LoDE_hstep 0.002 --LoDE_rstep 5.0
```

## Handover in simulation
The simulation takes a container ID and file ID from the CCM dataset as input. (Otherwise use the `--dataset` flag to indicate whether to process `train` or `test` or `all` recordings in the CCM dataset)

Acceptable flags:
- render - Render GUI of the simulation
- show_recording - Replay recording from real environment on the side
- record_sim - Record video of the simulated experiment
- run_desc - Name of the run
- res_path - path to save the results
- container_id - Provide CCM container ID to run on a single container
- recording_id - Provide CCM recording ID to run on a single recording
- dataset - Dataset to evaluate: train, test or all (if not processing a single recording)
- estimated_filling_mass - Use estimate of filling mass from the perception pipeline as input to the robot control
- safe_grasp - Enable safe grasping by the robot

Example:
```
python evaluate_handover.py --render --show_recording --run_desc test --dataset train
```
Numerical results are saved in the `results/` folder.

## Troubleshooting
**Q: My simulation is rendering very slowly in GUI mode. What can I do to speed it up?**

A: Try upgrading the drivers for your graphics card to the latest version. If you are using an Nvidia graphics card, try turning off "Sync to VBlank" under the OpenGL settings in the Nvidia settings panel.

## Acknowledgement

Plain citation
```
Pang, Y. L. and Xompero, A. and Oh, C. and Cavallaro, A., "Towards Safe Human-to-robot Handovers of Unknown Containers", 
in IEEE International Conference on Robot and Human Interactive Communication, Virtual, 8-12 August 2021.
```

BibTeX citation:
```
@InProceeding{Pang2021RO-MAN,
  AUTHOR     = {Pang, Y. L. and Xompero, A. and Oh, C. and Cavallaro, A.},
  TITLE      = {Towards Safe Human-to-robot Handovers of Unknown Containers},
  BOOKTITLE  = {IEEE International Conference on Robot and Human Interactive Communication},
  ADDRESS    = {Virtual},
  MONTH      = "8--12~" # AUG,
  YEAR       = 2021
}
```


## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, please contact y.l.pang@qmul.ac.uk or a.xompero@qmul.ac.uk. If you would like to file a bug report or a feature request, use the Github issue tracker.

## Licence

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
