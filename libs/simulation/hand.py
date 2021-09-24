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
import math
import json

import numpy as np
import pybullet as p

from libs.simulation.utils import *

HAND_START_COLOUR = (0.749, 0.498, 0.247, 0.0)
HAND_MOVE_COLOUR = (0.749, 0.498, 0.247, 1.0)
SHOW_OPENPOSE_HAND = False
HAND_CONSTRAINT_FORCE = 50

class Hand(object):

    def __init__(self, mesh_path, trajectory, LR, max_frame, table_uids, cont_uid, other_hand_uid=None, debug=False):
        self.mesh_path = mesh_path
        self.LR = LR
        self.max_frame = max_frame
        self.table_uids = table_uids
        self.cont_uid = cont_uid
        self.other_hand_uid = other_hand_uid
        self.hand_visible = False
        self.debug = debug

        self.keypoints_trajectory = trajectory
        # Get initial hand pose
        if self.keypoints_trajectory is None:
            self.hand_base_pos = (0.0, 0.0, 0.0)
            self.hand_base_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            self.robot_grasp_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        else:
            frame_keypoints = self.keypoints_trajectory[0]
            self.hand_base_pos, self.hand_base_quat, self.robot_grasp_quat = self.getBasePose(frame_keypoints)
        self.uid, self.cid = self.loadMesh()

    def loadMesh(self):

        # Spawn subject hand
        uid = p.loadMJCF(self.mesh_path)[0]

        # Update colour and make transparent
        num_joints = p.getNumJoints(uid)
        for j in range(-1, num_joints):
            p.changeVisualShape(uid, j, rgbaColor=HAND_START_COLOUR)

        # Set initial hand pose
        p.resetBasePositionAndOrientation(uid, self.hand_base_pos, self.hand_base_quat)
        cid = p.createConstraint(uid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.hand_base_pos, p.getQuaternionFromEuler([0.0, 0.0, 0.0]))

        # Disable collision
        for i in range(-1, p.getNumJoints(uid)):
            # Between hand and container
            p.setCollisionFilterPair(uid, self.cont_uid, i, -1, 0)
            # Between hand and table
            for table_uid in self.table_uids:
                p.setCollisionFilterPair(uid, table_uid, i, -1, 0)
            # Between left and right hand
            if self.other_hand_uid is not None:
                for j in range(-1, p.getNumJoints(self.other_hand_uid)):
                    p.setCollisionFilterPair(uid, self.other_hand_uid, i, j, 0)

        return uid, cid

    def getBasePose(self, frame_keypoints):
        hand_base_pos = frame_keypoints[0]

        # Get directional vector
        hand_dirv_a = frame_keypoints[9] - hand_base_pos
        hand_dirv_b = frame_keypoints[13] - hand_base_pos
        hand_dirv = hand_dirv_a + hand_dirv_b
        hand_dirv_norm = np.linalg.norm(hand_dirv)
        hand_dirv = hand_dirv / hand_dirv_norm

        # Get rotational vector
        hand_wv_temp = frame_keypoints[5] - frame_keypoints[17]
        hand_wv_temp = hand_wv_temp / np.linalg.norm(hand_wv_temp)

        # Get up vector
        hand_upv = np.cross(hand_wv_temp, hand_dirv)
        hand_upv = hand_upv / np.linalg.norm(hand_upv)

        # Get final rotational vector
        hand_wv = -np.cross(hand_upv, hand_dirv)
        hand_wv = hand_wv / np.linalg.norm(hand_wv)

        # Covert to quaternion
        hand_base_quat = vec2quaternion(hand_wv, -hand_dirv, -hand_upv)
        robot_grasp_quat = vec2quaternion(-hand_dirv, hand_upv, -hand_wv)

        # Offset hand pos
        hand_base_pos = hand_base_pos + hand_dirv * (-0.0776452 + hand_dirv_norm - 0.15)

        return hand_base_pos, hand_base_quat, robot_grasp_quat

    def updatePose(self, replayFrameID, start_movement):
        # Get pose for current frame
        frame_keypoints = self.keypoints_trajectory[replayFrameID + 1]
        self.hand_base_pos, self.hand_base_quat, self.robot_grasp_quat = self.getBasePose(frame_keypoints)

        # Update movement constraint
        p.changeConstraint(self.cid, self.hand_base_pos, self.hand_base_quat, maxForce=HAND_CONSTRAINT_FORCE)

        # Set finger joint angles
        joint_keypoint_id = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        joint_adj_keypoint_id = []
        for i in joint_keypoint_id:
            if i in [5, 9, 13, 17]:
                joint_adj_keypoint_id.append([0, i, i + 1])
            else:
                joint_adj_keypoint_id.append([i - 1, i, i + 1])

        joint_angle_list = []
        for adj_keypoints in joint_adj_keypoint_id:
            p1 = frame_keypoints[adj_keypoints[0]]
            p2 = frame_keypoints[adj_keypoints[1]]
            p3 = frame_keypoints[adj_keypoints[2]]

            v1 = p1 - p2
            v2 = p3 - p2
            # Normalise
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)

            joint_angle_list.append(math.pi - np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        joint_keypoint_map_dict = {9: 1, 11: 2, 13: 3, 17: 5, 19: 6, 21: 7, 24: 9, 26: 10, 28: 11,
                                   32: 13, 34: 14, 36: 15, 40: 17, 42: 18, 44: 19}

        # Set finger joint angles
        if self.LR == 'L':
            for i in range(p.getNumJoints(self.uid)):
                if i == 7:
                    p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, -1.3)
                elif i in joint_keypoint_map_dict.keys():
                    if i in [9, 11, 13]:
                        p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, joint_angle_list[joint_keypoint_id.index(joint_keypoint_map_dict[i])])
                    else:
                        p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, -joint_angle_list[joint_keypoint_id.index(joint_keypoint_map_dict[i])])
                else:
                    p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, 0.0)
        elif self.LR == 'R':
            # set right hand finger joint angles
            for i in range(p.getNumJoints(self.uid)):
                if i == 7:
                    p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, 1.3)
                elif i in joint_keypoint_map_dict.keys():
                    p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, joint_angle_list[joint_keypoint_id.index(joint_keypoint_map_dict[i])])
                else:
                    p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, 0.0)


        # Update colour
        keypoints_above_table = False
        for i in range(21):
            if 0.44 > frame_keypoints[i][0] > 0.0 and 1.0 > frame_keypoints[i][1] > 0.03:
                keypoints_above_table = True
                break
        if start_movement:
            if keypoints_above_table and not(self.hand_visible):
                num_joints = p.getNumJoints(self.uid)
                for j in range(-1, num_joints):
                    p.changeVisualShape(self.uid, j, rgbaColor=HAND_MOVE_COLOUR)
                self.hand_visible = True
