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

import os
import copy
import csv
import json
import pickle
import time
import math

import gym
import numpy as np
import scipy
import scipy.signal
import pybullet as p
import pybullet_data
from gym import spaces
from gym.utils import seeding

from libs.simulation.utils import *
from libs.simulation.ur5 import ur5
from libs.simulation.container import Container
from libs.simulation.hand import Hand

projectionMatrix = p.computeProjectionMatrixFOV(fov=90, aspect=1.777777778, nearVal=0.01, farVal=10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL

meshPath = os.getcwd() + "/data/meshes/objects/"
MPLPath = os.getcwd() + "/data/meshes/MPL/MPL.xml"


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def gripper_camera(obs):
    # gripper pos and ori
    pos = obs[-7:-4]
    ori = obs[-4:]  # last 4

    rotation = list(p.getEulerFromQuaternion(ori))
    rotation[0] = rotation[0] - math.pi * 0.5
    ori = p.getQuaternionFromEuler(rotation)
    rot_matrix = p.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # Initial vectors
    init_camera_vector = (0, 0, 1)  # z-axis
    init_up_vector = (0, 1, 0)  # y-axis

    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)

    view_matrix_gripper = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)

    img = p.getCameraImage(360, 360, view_matrix_gripper, projectionMatrix, shadow=0, renderer=image_renderer)
    rgba_img = img[2]
    rgb_img = rgba2rgb(rgba_img)

    return rgb_img


def render_side_cam(cam):
    tvec = cam['extrinsic']['tvec']
    rot_matrix = cam['extrinsic']['rvec']

    # Convert from camera to world coordinates
    pos = -rot_matrix.T.dot(tvec)
    rot_matrix = rot_matrix.T

    pos = pos.T[0] + np.array([0, 0, 1])
    pos = pos.tolist()

    # Initial vectors
    init_camera_vector = (0, 0, 1)  # z-axis
    init_up_vector = (0, 0, 1)  # y-axis

    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)

    view_matrix_sidecam = p.computeViewMatrix(pos, camera_vector, up_vector)

    img = p.getCameraImage(1280, 720, view_matrix_sidecam, projectionMatrix, shadow=0, renderer=image_renderer)

    rgba_img = img[2]
    rgb_img = rgba2rgb(rgba_img)

    return rgb_img


class handoverEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=True,
                 containerID=1,
                 recordingID='s0_fi0_fu0_b0_l0',
                 objectURDF='',
                 pred_tolerance=0.3):

        prGreen("Initialising environment...")

        self._timeStep = 1. / 240.  # simulation time step size
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 1280
        self._height = 720
        self.terminated = 0
        self._p = p

        self.containerID = containerID
        self.recordingID = recordingID

        self.container_mesh_path = './data/meshes/objects/CORSMAL_containers/' + str(self.containerID) + '/best.urdf'
        self.container_trajectory_path = './data/vision_estimations/container/' + str(self.containerID) + '/volume_estimation/' + str(self.recordingID) + '_properties.txt'

        self.container_trajectory = np.zeros([1, 3])
        self.lefthand_trajectory = None
        self.righthand_trajectory = None

        self.sim_mass = 0.01
        self.start_frame = 0
        self.handover_hand = 'L'

        self.replayFrameID = 0
        self.replayFinished = False

        self.pred_tolerance = pred_tolerance

        self.fail_contact = 0
        self.fail_weight = 0
        self.fail_width = 0
        self.stable_grasp = 0
        self.delivery_success = 0

        self.hand_visible = False

        # TODO: use real camera calib files for each recording
        self.cam_side_1 = self.readCalibration('./data/calibration/examples/calib/c1_calib.pickle')
        self.cam_side_2 = self.readCalibration('./data/calibration/examples/calib/c2_calib.pickle')
        self.cam_robot = self.readCalibration('./data/calibration/examples/calib/c3_calib.pickle')

        self.mesh_recon_frame = 0

        self.lefthand_debug_lines = []
        self.righthand_debug_lines = []

        # Render options
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Setup environment
        self._seed()
        self._reset()
        observationDim = len(self.getSceneObservation())
        observation_high = np.array([np.finfo(np.float32).max] * observationDim)

        action_dim = 4
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self._height, self._width), dtype=np.uint8)
        self.viewer = None

    def _reset(self):

        prGreen("Resetting environment...")

        self.terminated = 0
        self._envStepCounter = 0
        self.replayFrameID = 0
        self.replayFinished = False
        self.start_movement = False
        self.extendReplay = False
        self.gripper_state = 0.085
        self.target_gripper_width = 0.085
        self.joint_limit_effort = 0.01
        self.grasp_target_z = 0.0
        self.stable_grasp = 0
        self.delivery_success = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # Load scene setup
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        self.tables = []
        self.tables.append(p.loadURDF((os.path.join(meshPath,"table/table.urdf")),[0.0,0.0,0.0],p.getQuaternionFromEuler([0,0,0]),useFixedBase=True))
        self.tables.append(p.loadURDF((os.path.join(meshPath,"table/table.urdf")),[0.97,1.3,0.0],p.getQuaternionFromEuler([0,0,math.pi/2.0]),useFixedBase=True))
        target = p.loadURDF(os.path.join(meshPath, "target/target.urdf"), (0.75, 1.52, 0.625), p.getQuaternionFromEuler([0, 0, 0]))

        # Load robot
        self._arm = ur5(robotStartPos=[0.22, 1.02, 0.53], maxGripperForce=0.1)

        # Load container
        self.container_mesh_path = './data/meshes/objects/CORSMAL_containers/' + str(self.containerID) + '/best.urdf'
        self.container = Container(self.container_mesh_path, self.container_trajectory, self.sim_mass, self.tables)

        # Load hand models
        self.left_hand = Hand(MPLPath, self.lefthand_trajectory, 'L', len(self.container.trajectory), self.tables, self.container.uid)
        self.right_hand = Hand(MPLPath, self.righthand_trajectory, 'R', len(self.container.trajectory), self.tables, self.container.uid, other_hand_uid=self.left_hand.uid)

        p.stepSimulation()
        self._observation = self.getSceneObservation()
        
        return np.array(self._observation, dtype=object)

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Read calibration file for the chosen setup 
    def readCalibration(self, filepath):

        cam = dict.fromkeys(['intrinsic', 'extrinsic']);

        with open(filepath, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            calibration = u.load()

            cam['intrinsic'] = calibration[0]['rgb']
            cam['extrinsic'] = calibration[1]['rgb']

        return cam

    def getSceneObservation(self):
        self._observation = self._arm.getObservation()
        # scene_obs = get_scene_observation(self.tables)
        grip_img = gripper_camera(self._observation)
        # obs = [self._observation, scene_obs]

        cam_side_1_img = render_side_cam(self.cam_side_1)
        cam_side_2_img = render_side_cam(self.cam_side_2)
        cam_robot_img = render_side_cam(self.cam_robot)

        return grip_img, cam_side_1_img, cam_side_2_img, cam_robot_img

    def step(self, action, distance_to_target):
        # replay movement of container human manipulation
        if self.replayFrameID + 1 < len(self.container.trajectory) and (distance_to_target > 0.01):

            self.left_hand.updatePose(self.replayFrameID, self.start_movement)
            self.right_hand.updatePose(self.replayFrameID, self.start_movement)

            if not(self.start_movement):
                self.grasp_quat = list(p.getQuaternionFromEuler([math.pi, 0.0, -math.pi / 2.0]))
                cont_quat = p.getQuaternionFromEuler([0, -math.pi/2.0, 0])
            else:
                if self.handover_hand == 'L':
                    self.grasp_quat = self.left_hand.robot_grasp_quat
                    cont_quat = self.left_hand.hand_base_quat
                elif self.handover_hand == 'R':
                    self.grasp_quat = self.right_hand.robot_grasp_quat
                    cont_quat = self.right_hand.hand_base_quat

            self.container.updatePose(self.replayFrameID, cont_quat)

            self.replayFrameID += 1

        else:
            prYellow('Handover phase...')
            target_gripper_width = self.target_gripper_width

            state = p.getLinkState(self._arm.uid, 7)
            actualEndEffectorPos = np.asarray(state[0])
            last_target = actualEndEffectorPos + np.asarray(action[0:3])

            action[3] = target_gripper_width

            self._arm.maxGripperForce = self.joint_limit_effort
            self.replayFinished = True


        dv = 1.5
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        f = action[3]
        realAction = [dx, dy, dz, f]


        if self.replayFinished:
            self.start_movement = True
        else:
            if self.replayFrameID == self.start_frame:
                self.start_movement = True
                prYellow('Human manouevring phase...')

        self._arm.action(realAction, self.grasp_quat, self.start_movement, False)

        info = {}
        done = False
        reward = 0.0
        if not (self.replayFinished):
            for _ in range(8):
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1
        else:

            if self.replayFrameID + 1 == len(self.container.trajectory):
                for _ in range(480):
                    state = p.getLinkState(self._arm.uid, 7)
                    actualEndEffectorPos = np.asarray(state[0])
                    direction = last_target - actualEndEffectorPos
                    direction = direction * 1.5

                    realAction = list(direction) + [0.085]

                    self._arm.action(realAction, self.grasp_quat, self.start_movement, False)

                    p.stepSimulation()
                    if self._renders:
                        time.sleep(self._timeStep)
                    self._envStepCounter += 1

            # re-enable collision between container and table
            for col_objs in self.tables:
                p.setCollisionFilterPair(col_objs, self.container.uid, -1, -1, 1)

            # execute grasp
            realAction = [0.0, 0.0, 0.0, target_gripper_width]
            self._arm.action(realAction, self.grasp_quat, self.start_movement, False)
            for _ in range(100):
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1

            p.removeConstraint(self.container.cid)

            # check normal forces on container immediately after grasp executed
            total_normal_force_container_immediate, total_normal_force_hand_immediate = self.checkGraspNormalForce()

            # check actual gripper width
            actual_gripper_width = self.checkGraspActualWidth()

            for _ in range(100):
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1

            # evaluate grasp safety
            closest_l_left = self.checkGraspSafety(self._arm.uid, self.left_hand.uid)
            closest_l_right = self.checkGraspSafety(self._arm.uid, self.right_hand.uid)
            closest_l = min(closest_l_left, closest_l_right)

            # check final normal forces applied on container
            total_normal_force_container, total_normal_force_hand = self.checkGraspNormalForce()

            # remove hand
            p.removeBody(self.left_hand.uid)
            p.removeBody(self.right_hand.uid)
            p.removeAllUserDebugItems()

            # evaluate container delivery
            prYellow('Robot manouevring phase...')
            # slow down arm movement for delivery to prevent slipping
            self._arm.maxVelMultiplier = 0.3
            delivery_z = 0.775
            for _ in range(1000):
                delivery_pos = [0.75 - 0.14, 1.52, delivery_z]
                state = p.getLinkState(self._arm.uid, self._arm.endEffectorIndex)
                real_pos = list(state[0])
                # move arm
                dv = 1.0
                dx = (delivery_pos[0] - real_pos[0]) * dv
                dy = (delivery_pos[1] - real_pos[1]) * dv
                dz = (delivery_pos[2] - real_pos[2]) * dv
                f = target_gripper_width
                realAction = [dx, dy, dz, f]

                self._arm.action(realAction, _, True, True)

                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1
            # Lower object
            for _ in range(240 * 2):
                delivery_pos = [0.75 - 0.14, 1.52, 0.68]
                state = p.getLinkState(self._arm.uid, self._arm.endEffectorIndex)
                real_pos = list(state[0])
                # move arm
                dv = 1.0
                dx = (delivery_pos[0] - real_pos[0]) * dv
                dy = (delivery_pos[1] - real_pos[1]) * dv
                dz = (delivery_pos[2] - real_pos[2]) * dv
                f = target_gripper_width
                realAction = [dx, dy, dz, f]

                self._arm.action(realAction, _, True, True)

                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1
            # Open gripper
            for _ in range(100):
                delivery_pos = [0.75 - 0.14, 1.52, 0.68]
                state = p.getLinkState(self._arm.uid, self._arm.endEffectorIndex)
                real_pos = list(state[0])
                # move arm
                dv = 1.0
                dx = (delivery_pos[0] - real_pos[0]) * dv
                dy = (delivery_pos[1] - real_pos[1]) * dv
                dz = (delivery_pos[2] - real_pos[2]) * dv
                f = 0.085
                realAction = [dx, dy, dz, f]

                self._arm.action(realAction, _, True, True)

                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._envStepCounter += 1

            delivery_success, delivery_distance, delivery_beta = self.checkDeliverySuccess()

            done = True
            info = {'closest_l': closest_l, 'delivery_distance': delivery_distance, 'delivery_beta': delivery_beta, \
                    'total_normal_force_hand': total_normal_force_hand, 'total_normal_force_container': total_normal_force_container, \
                    'total_normal_force_container_immediate': total_normal_force_container_immediate, 'total_normal_force_hand_immediate': total_normal_force_hand_immediate, \
                    'actual_gripper_width': actual_gripper_width}

        return np.array(self._observation, dtype=object), reward, done, info


    def checkGraspSafety(self, robot_object_id, hand_object_id):

        max_l = 1.0
        closest_points = p.getClosestPoints(robot_object_id, hand_object_id, max_l)
        if len(closest_points) > 0:
            min_l = max_l
            for cp in closest_points:
                if cp[8] < min_l:
                    min_l = cp[8]

            min_l = max(0.0, min_l)
        else:
            min_l = max_l

        return min_l

    def checkGraspNormalForce(self):
        contact_points_container_robot = p.getContactPoints(self._arm.uid, self.container.uid)
        total_normal_force_container = 0.0
        for cp in contact_points_container_robot:
            total_normal_force_container += abs(cp[9])

        contact_points_lefthand_robot = p.getContactPoints(self._arm.uid, self.left_hand.uid)
        total_normal_force_lefthand = 0.0
        for cp in contact_points_lefthand_robot:
            total_normal_force_lefthand += abs(cp[9])
        contact_points_righthand_robot = p.getContactPoints(self._arm.uid, self.right_hand.uid)
        total_normal_force_righthand = 0.0
        for cp in contact_points_righthand_robot:
            total_normal_force_righthand += abs(cp[9])

        total_normal_force_hand = total_normal_force_lefthand + total_normal_force_righthand

        return total_normal_force_container, total_normal_force_hand


    def checkGraspActualWidth(self):
        js = p.getJointState(self._arm.uid, 14)
        # gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        gripper_opening_length = math.sin(0.715 - js[0]) * 0.1143 + 0.010
        return gripper_opening_length


    def checkDeliverySuccess(self):
        success = False
        state = p.getBasePositionAndOrientation(self.container.uid)
        pos = state[0]
        orn = state[1]
        rotation = list(p.getEulerFromQuaternion(orn))

        dir_x = math.cos(rotation[2]) * math.sin(rotation[1]) * math.cos(rotation[0]) + math.sin(rotation[2]) * math.sin(rotation[0])
        dir_y = math.sin(rotation[2]) * math.sin(rotation[1]) * math.cos(rotation[0]) - math.cos(rotation[2]) * math.sin(rotation[0])
        dir_z = math.cos(rotation[1]) * math.cos(rotation[0])

        dir_v = np.array([dir_x, dir_y, dir_z])
        unit_dir_v = dir_v / np.linalg.norm(dir_v)
        delivery_beta = np.arccos(np.clip(np.dot(unit_dir_v, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))


        target_pos = [0.75, 1.52, 0.625]
        gt_width_file = open('data/meshes/objects/CORSMAL_containers/' + str(self.containerID) + '/width.txt')
        next(gt_width_file)
        for line in gt_width_file:
            line_split = line.split(', ')
            if len(line_split) == 3:
                gt_cont_minz = float(line_split[1])
            break

        pos = list(np.asarray(pos)+np.asarray(unit_dir_v)*gt_cont_minz)
        delivery_distance = np.linalg.norm(np.array(pos) - np.array(target_pos))

        eta = 0.5
        if delivery_distance < eta and delivery_beta < math.pi / 18.0:
            delivery_success = (1.0 - (delivery_distance / eta))
        else:
            delivery_success = 0.0

        return delivery_success, delivery_distance, delivery_beta