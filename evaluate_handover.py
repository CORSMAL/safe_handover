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

import copy
import csv
import json
import time
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pybullet as p

from libs.simulation.utils import *
from libs.simulation.safe_grasp import *

from libs.simulation.handover_gym_env import handoverEnv


def getFilenames(containerID):
    filenames = []
    container_type_list = range(1, 7)
    filling_type_list = range(4)  # none, rice, pasta, water
    fullness_type_list = range(1, 3)  # 0, 0.5, 0.9
    background_type_list = range(1)
    lighting_type_list = range(1)
    scenario_type_list = range(2)

    if containerID in container_type_list:
        for filling in filling_type_list:
            if filling == 0:
                fullness = 0
                for background in background_type_list:
                    for lighting in lighting_type_list:
                        for scenario in scenario_type_list:
                            filenames.append('s' + str(scenario) + '_fi' + str(filling) + '_fu' + str(fullness) + '_b' + str(background) + '_l' + str(lighting))
            else:
                for fullness in fullness_type_list:
                    for background in background_type_list:
                        for lighting in lighting_type_list:
                            for scenario in scenario_type_list:
                                filenames.append('s' + str(scenario) + '_fi' + str(filling) + '_fu' + str(fullness) + '_b' + str(background) + '_l' + str(lighting))
    else:
        listimages = pd.read_csv('data/annotations/annotation_public_test.csv', sep=',')
        boolArray = listimages['Container ID'].values == containerID

        for j in listimages['Sequence'].values[boolArray]:
            if containerID == 11:
                if '{:04d}'.format(j) in ['0028']:
                    # missing openpose files
                    continue
            filenames.append('{:04d}'.format(j))

    return filenames


def loadGroundTruthTraining():

    # Training set
    container_list = ['Red cup', 'Small white cup', 'Small transparent cup', 'Green Glass', 'Wine Glass', 'Champagne Flute', 'Cereal Box', 'Biscuits box', 'Tea box']
    filling_type_list = ['Empty', 'Pasta', 'Rice', 'Water']
    filling_level_list = ['0', '50', '90']

    gsc_dict = {}
    csv_file = open(os.getcwd() + '/data/annotations/Annotation_Training.csv')
    csv_reader = csv.reader(csv_file, delimiter=',')
    # skip header
    next(csv_reader)

    for row in csv_reader:
        container_id = int(container_list.index(row[0]) + 1)
        filling_type = int(filling_type_list.index(row[1]))
        filling_level = int(filling_level_list.index(row[2]))

        if container_id not in gsc_dict:
            gsc_dict[container_id] = {}
        if filling_type not in gsc_dict[container_id]:
            gsc_dict[container_id][filling_type] = {}
        if filling_level not in gsc_dict[container_id][filling_type]:
            gsc_dict[container_id][filling_type][filling_level] = {}

        gsc_dict[container_id][filling_type][filling_level]['container_capacity'] = float(row[3]) if row[3] != '' else None
        gsc_dict[container_id][filling_type][filling_level]['container_mass'] = float(row[4]) if row[4] != '' else None
        gsc_dict[container_id][filling_type][filling_level]['filling_mass'] = float(row[5]) if row[5] != '' else None
        # gsc_dict[container_id][filling_type][filling_level]['container_width'] = float(row[6]) if row[6] != '' else None

    return gsc_dict

def loadGroundTruthTesting():

    # Public test set
    gsc_dict_public_test = {}
    public_test_csv = open(os.getcwd() + '/data/annotations/annotation_public_test.csv')
    csv_reader = csv.reader(public_test_csv, delimiter=',')
    # skip header
    next(csv_reader)

    for row in csv_reader:
        container_id = int(row[0])
        sequence = int(row[1])

        if container_id not in gsc_dict_public_test:
            gsc_dict_public_test[container_id] = {}
        if sequence not in gsc_dict_public_test[container_id]:
            gsc_dict_public_test[container_id][sequence] = {}

        gsc_dict_public_test[container_id][sequence]['container_capacity'] = float(row[4])
        gsc_dict_public_test[container_id][sequence]['container_mass'] = float(row[6])
        gsc_dict_public_test[container_id][sequence]['filling_mass'] = float(row[5])
        # gsc_dict_public_test[container_id][sequence]['container_width'] = float(row[11])

    return gsc_dict_public_test

def getPublicTestLabels():
    public_test_annotation = {}
    public_test_annotation_file = open('data/annotations/annotation_public_test.csv')
    next(public_test_annotation_file)
    for line in public_test_annotation_file:
        line_split = line.split(',')
        containerID = int(line_split[0])
        sequence = int(line_split[1])
        filling_level = [0, 50, 90].index(int(line_split[2]))
        filling_type = int(line_split[3])
        scenario = int(line_split[7][1])
        background = int(line_split[8][1])
        lighting = int(line_split[9][1])

        if containerID not in public_test_annotation:
            public_test_annotation[containerID] = {}
        if sequence not in public_test_annotation[containerID]:
            public_test_annotation[containerID][sequence] = {}

        public_test_annotation[containerID][sequence]['filling_level'] = filling_level
        public_test_annotation[containerID][sequence]['filling_type'] = filling_type
        public_test_annotation[containerID][sequence]['scenario'] = scenario
        public_test_annotation[containerID][sequence]['background'] = background
        public_test_annotation[containerID][sequence]['lighting'] = lighting

    return public_test_annotation

def loadGroundTruth(real_data_training, real_data_testing, public_test_labels, container_id, recording_id):
    prGreen('Loading ground truths...')
    file_labels = {}
    real_data = {}
    if container_id in [1,2,3,4,5,6]:
        file_labels['scenario'] = int(recording_id[1:2])
        file_labels['filling_type'] = int(recording_id[5:6])
        file_labels['filling_level'] = int(recording_id[9:10])
        file_labels['background'] = int(recording_id[12:13])
        file_labels['lighting'] = int(recording_id[15:16])

        real_data['container_capacity'] = real_data_training[container_id][file_labels['filling_type']][file_labels['filling_level']]['container_capacity']
        real_data['container_mass'] = real_data_training[container_id][file_labels['filling_type']][file_labels['filling_level']]['container_mass']
        real_data['filling_mass'] = real_data_training[container_id][file_labels['filling_type']][file_labels['filling_level']]['filling_mass']
        # real_data['container_width'] = real_data_training[container_id][file_labels['filling_type']][file_labels['filling_level']]['container_width']/1000.0
    elif container_id in [10,11]:
        annotation = public_test_labels[container_id][int(recording_id)]
        file_labels['scenario'] = annotation['scenario']
        file_labels['filling_type'] = annotation['filling_type']
        file_labels['filling_level'] = annotation['filling_level']
        file_labels['background'] = annotation['background']
        file_labels['lighting'] = annotation['lighting']

        real_data['container_capacity'] = real_data_testing[container_id][int(recording_id)]['container_capacity']
        real_data['container_mass'] = real_data_testing[container_id][int(recording_id)]['container_mass']
        real_data['filling_mass'] = real_data_testing[container_id][int(recording_id)]['filling_mass']
        # real_data['container_width'] = real_data_testing[container_id][int(recording_id)]['container_width']/1000.0

    try:
        width_file = open('data/meshes/objects/CORSMAL_containers/' + str(container_id) + '/width.txt')
    except FileNotFoundError:
        return False, file_labels, real_data

    real_height_width = []
    next(width_file)
    min_z = 0.0
    max_z = 0.0
    first = True
    for line in width_file:
        line_split = line.split(', ')
        if len(line_split) == 3:
            z_pos = float(line_split[1])
            radius = float(line_split[2][:-2])
            if first:
                min_z = z_pos
                max_z = z_pos
                first = False
            else:
                if z_pos < min_z:
                    min_z = z_pos
                if z_pos > max_z:
                    max_z = z_pos
            real_height_width.append([z_pos, radius * 2.0])
    real_data['width_per_height_interval'] = real_height_width
    real_data['min_z_container_offset'] = min_z
    real_data['max_z_container_offset'] = max_z

    return True, file_labels, real_data


def loadContainerTrajectory(new_CCM_fname, dataset):
    measured_traj = []
    trajectory_path = './data/CCM/vision_estimations/' + dataset + '/volume_estimation/' + new_CCM_fname + '_properties.txt'
    file = open(trajectory_path)
    next(file)
    for line in file:
        pos = [float(i) for i in line.split(', ')[1:4]]
        measured_traj.append([pos[0], pos[1], pos[2] + TRAJECTORY_Z_OFFSET])

    filtered_traj = kalmanFilter(measured_traj)

    return np.asarray(filtered_traj)


def loadHandTrajectory(container_id, recording_id, max_frame, LR):
    trajectory_path = os.path.join('data/vision_estimations/openpose/', str(container_id), str(recording_id))
    keypoints_trajectory_measured = np.zeros([21, 3, max_frame])
    keypoints_trajectory_filtered = np.zeros([21, 3, max_frame])

    for replayFrameID in range(max_frame):
        with open(os.path.join(trajectory_path, '{0:012d}_keypoints.json'.format(replayFrameID)), 'r') as hand_pose_file:
            hand_pose_data = json.load(hand_pose_file)

            if LR == 'L':
                keypoints_file = hand_pose_data['people']['hand_left_keypoints_3d']
            elif LR == 'R':
                keypoints_file = hand_pose_data['people']['hand_right_keypoints_3d']

            # Get measured trajectory of all keypoints
            for j in range(21):
                keypoint_pos = np.asarray(keypoints_file[3 * j:3 * j + 3])
                keypoint_pos[2] = keypoint_pos[2] + TRAJECTORY_Z_OFFSET
                keypoints_trajectory_measured[j, :, replayFrameID] = keypoint_pos

    # Apply Kalman fiter over the entire trajectory for each keypoint
    for k in range(3):
        for j in range(21):
            keypoints_trajectory_filtered[j, :, :] = kalmanFilter(keypoints_trajectory_measured[j].T).T

    # Collect all keypoints
    keypoints_trajectory = []
    for i in range(max_frame):
        frame_keypoints = []
        for j in range(21):
            frame_keypoints.append([keypoints_trajectory_filtered[j, 0, i], keypoints_trajectory_filtered[j, 1, i], keypoints_trajectory_filtered[j, 2, i]])
        keypoints_trajectory.append(frame_keypoints)

    return np.asarray(keypoints_trajectory)


def loadPrediction(container_id, recording_id, new_CCM_fname, dataset, CORSMAL_challenge):
    success = True
    pred_data = {}

    filling_density_list = [0.0, 0.41, 0.85, 1.0, 0.0]
    filling_level_list = [0.0, 0.5, 0.9, 0.0]

    pred_traj = []
    pred_width_list = []
    pred_mass_list = []
    pred_filling_type_list = []
    pred_filling_level_list = []
    try:
        container_estimations_path = './data/CCM/vision_estimations/' + dataset + '/volume_estimation/'
        properties_file = open(os.path.join(container_estimations_path, new_CCM_fname+'_properties.txt'))
        width_file = open(os.path.join(container_estimations_path, new_CCM_fname+'_width.txt'))
        if CORSMAL_challenge:
            corsmal_sub_file = open('./data/CORSMAL_challenge_submissions/'+method+'.csv')
    except FileNotFoundError:
        success = False
        return success, None

    prGreen('Loading predictions...')
    next(properties_file)
    # Load CORSMAL challenge submission predictions
    if CORSMAL_challenge:
        corsmal_csv_reader = csv.reader(corsmal_sub_file, delimiter=',')
        next(corsmal_csv_reader, None)
        pred_filling_level_id = 0
        pred_filling_type = 0
        for row in corsmal_csv_reader:
            if int(container_id) == int(row[0]) and int(recording_id) == int(row[1]):
                if method == 'bit2':
                    pred_data['container_capacity'] = float(row[4])
                    pred_filling_type = int(row[2])
                    pred_filling_level_id = [0, 50, 90].index(int(row[3]))
                else:
                    pred_data['container_capacity'] = float(row[2])
                    pred_filling_type = int(row[3])
                    pred_filling_level_id = [0, 50, 90].index(int(row[4]))
    
    # Load predictions
    pred_data['container_capacity'] = 0.0
    curr_frame = -1
    prev_frame = -1
    for line_id, line in enumerate(properties_file):
        # Vision pipeline failed, skip
        if 'Traceback' in line:
            success = False
            return success, None

        curr_frame = int(line.split(', ')[0])

        # Bug in vision pipeline output with multiple lines of same frame, skip
        if curr_frame == prev_frame:
            success = False
            return success, None
        else:
            prev_frame = curr_frame

        # Load predictions
        pred_traj.append([float(line.split(', ')[1]), float(line.split(', ')[2]), float(line.split(', ')[3]) + 0.625])
        pred_width_list.append(float(line.split(', ')[7])/1000.0)

        if not(CORSMAL_challenge):
            # Container capacity estimated at the first frame
            if line_id == 1:
                pred_data['container_capacity'] = float(line.split(', ')[9])

            pred_filling_type = int(line.split(', ')[11])
            pred_filling_level_id = int(line.split(', ')[10])

        # Filling type
        pred_filling_type_list.append(pred_filling_type)

        # Filling level
        pred_filling_level = filling_level_list[pred_filling_level_id]
        pred_filling_level_list.append(pred_filling_level)

        # Filling mass
        pred_density = filling_density_list[pred_filling_type]
        pred_mass_list.append(pred_data['container_capacity'] * pred_filling_level * pred_density)

    # Apply kalman filter to trajectory
    traj = kalmanFilter(pred_traj)
    pred_data['container_trajectory'] = traj
    pred_data['container_width'] = pred_width_list
    pred_data['filling_mass'] = pred_mass_list
    pred_data['filling_type'] = pred_filling_type_list
    pred_data['filling_level'] = pred_filling_level_list


    # Get the width of the container at each height interval
    pred_height_width = []
    next(width_file)
    min_z = 0.0
    max_z = 0.0
    max_width = 0.0
    first = True
    for line in width_file:
        line_split = line.split(', ')
        if len(line_split) == 3:
            z_pos = float(line_split[1])
            radius = float(line_split[2][:-2])
            if first:
                min_z = z_pos
                max_z = z_pos
                first = False
            else:
                if z_pos < min_z:
                    min_z = z_pos
                if z_pos > max_z:
                    max_z = z_pos
            pred_height_width.append([z_pos, radius * 2.0])
    pred_data['width_per_height_interval'] = pred_height_width
    pred_data['min_z_container_offset'] = min_z
    pred_data['max_z_container_offset'] = max_z

    return success, pred_data


def loadStartFrame(containerID, recordingID):
    f = open('data/annotations/handover_start_annotation.csv')
    csv_reader = csv.reader(f, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        if containerID in [1,2,3,4,5,6]:
            if containerID == int(row[0]) and recordingID == row[1]:
                return int(row[2]), str(row[3])
        else:
            if containerID == int(row[0]) and str(int(recordingID)) == row[1]:
                return int(row[2]), str(row[3])


def loadCCMMapping():
    CCM_mapping_dict = {}
    CCM_mapping_dict_inv = {}
    for i in range(1,10):
        CCM_mapping_dict[i] = {}
    with open('data/CCM/ccm_annotations_train_set_mapping.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            container_id = int(l[0])
            old_fname = str(l[2:18])
            new_fname = int(l[21:])
            CCM_mapping_dict[container_id][old_fname] = new_fname
            CCM_mapping_dict_inv[new_fname] = [container_id, old_fname]

    return CCM_mapping_dict, CCM_mapping_dict_inv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulated handover experiments')

    parser.add_argument('--env', default='handover_gym_env')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode')
    parser.add_argument('--render', dest='render', action='store_true', help='Render GUI of the simulation')
    parser.add_argument('--show_recording', dest='show_recording', action='store_true', help='Replay recording from real environment')
    parser.add_argument('--record_sim', dest='record_sim', action='store_true', help='Record video of the simulated experiment')
    
    parser.add_argument('--run_desc', default='baseline_A', type=str, help='Name of the run')
    parser.add_argument('--CORSMAL_challenge', dest='CORSMAL_challenge', action='store_true', help='Indicate if the method is a submission of the CORSMAL challenge')
    parser.add_argument('--results_path', default='results/', type=str, help='Path to results file')
    parser.add_argument('--dataset', default='train', type=str, help='Dataset to evaluate: train, test or all')
    parser.add_argument('--run_repeat', default=1, type=int, help='How many times to evaluate for each recording')
    parser.add_argument('--recording_id', type=int, help='Run on single recording')
    
    parser.add_argument('--gripper_open_width', default=0.085, type=float, help='Width of gripper when opened')

    parser.add_argument('--estimated_filling_mass', default=1, type=int, help='Use estimate of filling mass from the perception pipeline')
    parser.add_argument('--safe_grasp', default=1, type=int, help='Enable safe grasping')

    args = parser.parse_args()

    prCyan('Starting run: %s' % args.run_desc)
    prCyan('Loading gym environment: %s' % args.env)

    # gym_env = __import__('utils.gym_envs.%s' % args.env, fromlist=['utils.gym_envs'])
    gym_env = __import__(args.env)
    env = gym_env.handoverEnv(renders=args.render)
    p.resetDebugVisualizerCamera(1.20, 151.6, -19.4,[0.10, -0.04, 0.72])

    dataset = args.dataset
    CCM_mapping_dict, CCM_mapping_dict_inv = loadCCMMapping()
    if args.recording_id:
        container_id, recording_id = CCM_mapping_dict_inv[int(args.recording_id)]
        container_list = [container_id]
        recording_list = [[recording_id]]
        if container_id in [1,2,3,4,5,6]:
            dataset = 'train'
        elif container_id in [10, 11]:
            dataset = 'test'

    else:
        container_list = [1,2,3,4,5,6,10,11]
        recording_list = []
        if dataset == 'train':
            container_list = [1,2,3,4,5,6]
        elif dataset == 'test':
            container_list = [10,11]
        for container_id in container_list:
            recording_list.append(getFilenames(container_id))

    # Preload all file labels and ground truth data
    real_data_training = None
    real_data_testing = None
    public_test_labels = None
    if dataset in ['train', 'all']:
        real_data_training = loadGroundTruthTraining()
    if dataset in ['test', 'all']:
        real_data_testing = loadGroundTruthTesting()
        public_test_labels = getPublicTestLabels()

    for run in range(args.run_repeat):
        # Log the parameters used for this run
        results_dir_path = os.path.join(args.results_path, 'run{}'.format(run+1))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        if not(args.debug):
            experiment_log_file_path = os.path.join(results_dir_path, '{}_params.txt'.format(args.run_desc))
            with open(experiment_log_file_path, 'w') as logfile:
                for arg in vars(args):
                    logfile.write('{}: {}\n'.format(arg, getattr(args, arg)))
        results_file_path = os.path.join(results_dir_path, '{}.csv'.format(args.run_desc))
        results_file = open(results_file_path, 'w+', newline='')
        results_file_csvwriter = csv.writer(results_file, delimiter=',')
        results_file_csvwriter.writerow(['container_id', 'recording_id', 'scenario', 'filling_type', 'filling_level', 'background', 'lighting', 'fail_vision_pipeline',
                                        'pred_filling_type', 'pred_filling_level', 'pred_volume', 'gt_volume', 'max_frame', 'handover_frame', 'safe_grasp_available', 'closest_l', 'pred_width',
                                        'gt_width', 'target_gripper_width', 'actual_gripper_width', 'pred_mesh_width', 'gt_mesh_width',  'pred_weight', 'gt_weight', 'delivery_distance', 'delivery_angle', 'total_normal_force_container', 'total_normal_force_hand',
                                        'total_normal_force_container_immediate', 'total_normal_force_hand_immediate', 'grasp_maxz', 'grasp_minz'])
        results_file.close()

        for cidx, container_id in enumerate(container_list):
            for recording_id in recording_list[cidx]:
                if container_id in [1,2,3,4,5,6]:
                    dataset = 'train'
                elif container_id in [10,11]:
                    dataset = 'test_pub'
                new_CCM_fname = '{:06d}'.format(CCM_mapping_dict[container_id][recording_id])
                # Making sure OpenPose data exist for this recording...
                try:
                    hp_file = open('data/vision_estimations/openpose/' + str(container_id) + '/' + str(recording_id) + '/{0:012d}'.format(0) + '_keypoints.json', 'r')
                except FileNotFoundError:
                    prRed('Warning: Failed to load hand keypoint estimation files')
                    continue
                print("----------------------------------------")
                prCyan('Processing container:' + str(container_id) + ' recording:' + str(recording_id))

                # Open results log file csv writer
                results_file = open(results_file_path, 'a+', newline='')
                results_file_csvwriter = csv.writer(results_file, delimiter=',')

                # Load starting frame and handover hand annotations
                start_frame, handover_hand = loadStartFrame(container_id, recording_id)

                # Load file labels and ground truths for this recording
                load_file_succecss, file_labels, real_data = loadGroundTruth(real_data_training, real_data_testing, public_test_labels, container_id, recording_id)
                if not(load_file_succecss):
                    results_file_csvwriter.writerow([container_id, recording_id, file_labels['scenario'], file_labels['filling_type'], file_labels['filling_level'], file_labels['background'], file_labels['lighting'],
                                                    '1', '0', '0', '0.0', '0.0', '0', '0', '0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                                                    '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'])
                    results_file.close()
                    prRed('Warning: Failed to load ground truth files')
                    continue

                # Load container and hand trajectories
                container_trajectory = loadContainerTrajectory(new_CCM_fname, dataset)
                lefthand_trajectory = loadHandTrajectory(container_id, recording_id, len(container_trajectory), 'L')
                righthand_trajectory = loadHandTrajectory(container_id, recording_id, len(container_trajectory), 'R')
                
                # Load predictions
                load_file_succecss, pred_data = loadPrediction(container_id, recording_id, new_CCM_fname, dataset, args.CORSMAL_challenge)
                if not(load_file_succecss):
                    results_file_csvwriter.writerow([container_id, recording_id, file_labels['scenario'], file_labels['filling_type'], file_labels['filling_level'], file_labels['background'], file_labels['lighting'],
                                                    '1', '0', '0', '0.0', '0.0', '0', '0', '0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                                                    '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0'])
                    results_file.close()
                    prRed('Warning: Failed to load prediction files')
                    continue
                # Use ground truth where specified
                # if not(args.estimated_container_shape):
                #     pred_data['container_width'] = [real_data['container_width']] * len(container_trajectory)
                if not(args.estimated_filling_mass):
                    pred_data['container_capacity'] = real_data['container_capacity']
                    pred_data['filling_mass'] = [real_data['filling_mass']] * len(container_trajectory)
                    pred_data['filling_type'] = [file_labels['filling_type']] * len(container_trajectory)
                    pred_data['filling_level'] = [file_labels['filling_level']] * len(container_trajectory)

                # Reset simulation
                env.containerID = container_id
                env.recordingID = recording_id
                env.container_trajectory = container_trajectory
                env.lefthand_trajectory = lefthand_trajectory
                env.righthand_trajectory = righthand_trajectory
                env.sim_mass = (real_data['container_mass'] + real_data['filling_mass']) / 1000.0
                env.start_frame = start_frame
                env.handover_hand = handover_hand
                env._reset()
                
                # Record video directly
                if args.record_sim:
                    print('Start recording')
                    log_uid = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName='videos/' + str(containerID) + '/' + filename + '.mp4')
                # Display recording side by side
                if args.show_recording:
                    cap = cv2.VideoCapture('data/CCM/{}/view1/rgb/{}.mp4'.format(dataset, new_CCM_fname))

                direction = [0.0, 0.0, 0.0]
                safe_grasp_debug_lines = [[], []]
                for replayFrameID in range(len(pred_data['container_trajectory'])):
                    # Display current recorded frame from left camera view
                    if args.show_recording:
                        ret, frame = cap.read()
                        frame = cv2.resize(frame, (640, 360))
                        cv2.imshow('frame', frame)
                        cv2.moveWindow('frame', 0, 520)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # get current gripper position
                    state = p.getLinkState(env._arm.uid, 7)
                    actualEndEffectorPos = np.asarray(state[0])

                    safe_grasp_avail, grasp_pos, grasp_orn, safe_grasp_debug_lines = getSafeGraspPos(container_id, recording_id, replayFrameID, pred_data, safe_grasp_debug_lines, lefthand_trajectory, righthand_trajectory, start_frame, env.container.uid)
                    distance_to_target = 999.0
                    if replayFrameID > 0:
                        # Update grasping target at 30Hz (every 8 steps in simulation)
                        target = copy.deepcopy(pred_data['container_trajectory'][replayFrameID])
                        if args.safe_grasp:
                            # do not approach container if there is no safe grasp available
                            if safe_grasp_avail:
                                target += 0.15 * grasp_orn['dir_v']
                            else:
                                target += 0.25 * grasp_orn['dir_v']
                            # Adjust grasp z using graspable area
                            target += grasp_pos['target_z'] * grasp_orn['w_v']
                        else:
                            target += 0.15 * grasp_orn['dir_v']
                        direction = np.asarray(target - actualEndEffectorPos)
                        distance_to_target = np.linalg.norm(np.array(direction))
                        direction = list(direction)

                        # Calculate gripper width for grasping from predictions
                        z_offset = grasp_pos['target_z']
                        target_width = 0.0
                        for target_width_id, i in enumerate(pred_data['width_per_height_interval'][::-1]):
                            if z_offset >= i[0]:
                                target_width = i[1]
                                break
                        target_gripper_width = max(min(target_width, 0.085) - 0.02, 0.01)

                        # Get real target width
                        real_target_width = 0.0
                        for _, i in enumerate(real_data['width_per_height_interval'][::-1]):
                            if z_offset >= i[0]:
                                real_target_width = i[1] * 2.0
                                break

                        # Calculate gripper effort for grasping
                        pred_total_mass = pred_data['filling_mass'][replayFrameID] + real_data['container_mass']
                        # Pre-trained grasping model
                        F_c = (pred_total_mass / 1000.0) * (10.0 + 27.9)
                        a = 25.347238867884624
                        b = 0.04516976846125509
                        joint_limit_effort = max((F_c - b) / a, 0.01)

                        env.target_gripper_width = target_gripper_width
                        env.joint_limit_effort = joint_limit_effort

                    _, reward, done, info = env.step(direction + [args.gripper_open_width], distance_to_target)

                    if done:
                        break

                # Stop replay of recording
                if args.show_recording:
                    cap.release()
                    cv2.destroyAllWindows()

                # Save recording of simulation
                if args.record_sim:
                    print('End recording')
                    p.stopStateLogging(log_uid)
                results_file_csvwriter.writerow([container_id, recording_id, file_labels['scenario'], file_labels['filling_type'], file_labels['filling_level'], file_labels['background'], file_labels['lighting'],
                                                '0', pred_data['filling_type'][replayFrameID], pred_data['filling_level'][replayFrameID], pred_data['container_capacity'], real_data['container_capacity'], len(pred_data['container_trajectory']), replayFrameID, int(safe_grasp_avail), info['closest_l'],
                                                pred_data['container_width'][replayFrameID], 'n/a', target_gripper_width, info['actual_gripper_width'], target_width, real_target_width, pred_total_mass,
                                                real_data['container_mass']+real_data['filling_mass'], info['delivery_distance'], info['delivery_beta'], info['total_normal_force_container'], info['total_normal_force_hand'],
                                                info['total_normal_force_container_immediate'], info['total_normal_force_hand_immediate'], grasp_pos['max_z'], grasp_pos['min_z']])
                results_file.close()
                prPurple('Finished.')
