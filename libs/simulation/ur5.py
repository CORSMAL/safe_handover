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

# Based on https://github.com/sholtodouglas/ur5pybullet/blob/master/ur5.py

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import math
import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from itertools import chain
from collections import deque

import random
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import time
import itertools


def setup(p, uid):
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "robotiq_85_left_knuckle_joint"]
    mimicParentName = "robotiq_85_left_knuckle_joint"
    mimicChildren = ["robotiq_85_right_knuckle_joint",
                    "robotiq_85_left_inner_knuckle_joint",
                    "robotiq_85_right_inner_knuckle_joint",
                    "robotiq_85_left_finger_tip_joint",
                    "robotiq_85_right_finger_tip_joint"]
    mimic_multiplier = [1, 1, 1, -1, -1]

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    return joints, controlJoints, mimicParentName, mimicChildren, mimic_multiplier



class ur5:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, vr = False, robotStartPos=[0.0,0.0,-0.1], maxGripperForce=0.1):



        self.robotUrdfPath = "./data/meshes/ur5/ur5.urdf"
        self.robotStartPos = robotStartPos
        self.robotStartOrn = p.getQuaternionFromEuler([0.0,0.0,math.pi])

        self.maxGripperForce = maxGripperForce
        self.maxVelMultiplier = 1.0

        self.xin = self.robotStartPos[0]
        self.yin = self.robotStartPos[1]
        self.zin = self.robotStartPos[2]

        self.reset()


    def reset(self):
        
        self.uid = p.loadURDF(os.path.join(os.getcwd(),self.robotUrdfPath), self.robotStartPos, self.robotStartOrn, 
                             flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlJoints, self.mimicParentName, self.mimicChildren, self.mimic_multiplier = setup(p, self.uid)
        self.endEffectorIndex = 7 # ee_link
        self.numJoints = p.getNumJoints(self.uid)
        self.active_joint_ids = []
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self.active_joint_ids.append(joint.id)
        self.resetJointPoses()



    def getActionDimension(self):
        return 8  # position x,y,z and ori quat and finger angle
    def getObservationDimension(self):
        return len(self.getObservation())

    def setPosition(self, pos, quat):

        p.resetBasePositionAndOrientation(self.uid,pos,
                                          quat)

    def resetJointPoses(self, initJointPose=[-1.6730971990388346, -1.6598406519858835, 2.3176031148228584, -0.6869744035891363, 1.466861827095387, 1.2471890665968965e-06]):

        # move to this ideal init point
        for i, jp in enumerate(initJointPose):
            p.resetJointState(self.uid, i+1, jp)




    def getObservation(self):
        observation = []
        state = p.getLinkState(self.uid, self.endEffectorIndex, computeLinkVelocity = 1)
        pos = state[0]
        orn = state[1]
        

        observation.extend(list(pos))
        observation.extend(list(orn))

        joint_states = p.getJointStates(self.uid, self.active_joint_ids)
        
        joint_positions = list()
        joint_velocities = list()
        
        for joint in joint_states:
            
            joint_positions.append(joint[0])
            joint_velocities.append(joint[1])
            
        return joint_positions + joint_velocities + observation


    def action(self, motorCommands, target_quat, issue_action, delivery):

        if issue_action:
            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            
            state = p.getLinkState(self.uid, self.endEffectorIndex)
            pos = list(state[0])

            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy
            pos[2] = pos[2] + dz
        else:
            pos = [0.30, 0.65, 0.8]
        
        if delivery:
            orn = list(p.getQuaternionFromEuler([math.pi, 0.0, 0.0]))
        else:
            orn = target_quat
        jointPose = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, pos, orn)

        gripper_opening_length = motorCommands[3]
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        
        poses = []
        indexes = []
        maxForces = []
        maxVelocities = []

        # control ur5
        for i, name in enumerate(self.controlJoints):

            if name==self.mimicParentName:
                continue

            joint = self.joints[name]

            poses.append(jointPose[i])
            indexes.append(joint.id)
            maxForces.append(joint.maxForce)
            maxVelocities.append(joint.maxVelocity)
            p.setJointMotorControl2(self.uid, joint.id, p.POSITION_CONTROL,
                                    targetPosition=jointPose[i],
                                    positionGain=0.005,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity*self.maxVelMultiplier)

        # control gripper
        joint = self.joints[self.mimicParentName]
        p.setJointMotorControl2(self.uid,
                joint.id,
                p.POSITION_CONTROL,
                targetPosition=gripper_opening_angle,
                force=self.maxGripperForce,
                maxVelocity=joint.maxVelocity)
        for j in range(len(self.mimicChildren)):
            joint = self.joints[self.mimicChildren[j]]
            p.setJointMotorControl2(self.uid, joint.id, p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle * self.mimic_multiplier[j],
                                    force=self.maxGripperForce,
                                    maxVelocity=joint.maxVelocity)



    

    def move_to(self, position_delta):

        x = position_delta[0]
        y = position_delta[1]
        z = position_delta[2]
        
        orn = position_delta[3:7]
        gripper_opening_length = position_delta[7]
        
        jointPose = p.calculateInverseKinematics(self.uid, self.endEffectorIndex, [x,y,z], orn)

        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        
        self.action(jointPose, gripper_opening_angle)
        return jointPose

