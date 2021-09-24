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

import cv2
import numpy as np
import pandas as pd
import pybullet as p

from libs.simulation.utils import *


def getUnsafeGrasp(grasphand_keypoints, traj, max_rad, safety_offset, replayFrameID, up_v, dir_v, w_v):
    # get z of grasping hand keypoint pos
    grasphand_z = []
    p1 = traj[replayFrameID] - max_rad * 2.0 * up_v
    p2 = traj[replayFrameID] + max_rad * 2.0 * up_v
    u = p1 - p2
    x_min = np.dot(u, p2)
    x_max = np.dot(u, p1)

    p3 = traj[replayFrameID] - max_rad * dir_v
    p4 = traj[replayFrameID] + max_rad * dir_v
    v = p3 - p4
    y_min = np.dot(v, p4)
    y_max = np.dot(v, p3)

    p5 = traj[replayFrameID]
    p6 = traj[replayFrameID] + w_v
    w = p6 - p5
    w = w / np.linalg.norm(w)
    z_min = np.dot(w, p6)
    z_max = np.dot(w, p5)

    for i in range(21):
        if y_min < np.dot(v, grasphand_keypoints[i]) and x_min < np.dot(u, grasphand_keypoints[i]) < x_max:
            grasphand_z.append(np.dot(w, grasphand_keypoints[i]) - np.dot(w, traj[replayFrameID]))
    if len(grasphand_z) == 0:
        grasp_max = 0.0
        grasp_min = 0.0
    else:
        grasp_max = max(grasphand_z) + safety_offset
        grasp_min = min(grasphand_z) - safety_offset

    return grasp_max, grasp_min


def getSafeUnsafeDiff(grasp_unsafe_maxz, grasp_unsafe_minz, safe_interval):
    safe_interval_final = []
    if grasp_unsafe_maxz == grasp_unsafe_minz:
        return safe_interval
    for si in safe_interval:
        safe_interval_max = max(si)
        safe_interval_min = min(si)
        if grasp_unsafe_maxz > safe_interval_max and grasp_unsafe_minz < safe_interval_min:
            # hand cover entire container, no safe grasp available
            pass
        elif safe_interval_min > grasp_unsafe_maxz or grasp_unsafe_minz > safe_interval_max:
            # no overlap, all area graspable
            safe_interval_final.append(si)
        elif grasp_unsafe_maxz > safe_interval_min > grasp_unsafe_minz:
            # partial overlap, one area graspable
            safe_interval_final.append([grasp_unsafe_maxz, safe_interval_max])
        elif grasp_unsafe_maxz > safe_interval_max > grasp_unsafe_minz:
            # partial overlap, one area graspable
            safe_interval_final.append([safe_interval_min, grasp_unsafe_minz])
        elif grasp_unsafe_maxz < safe_interval_max and grasp_unsafe_minz > safe_interval_min:
            # full overlap, two area graspable
            safe_interval_final.append([safe_interval_min, grasp_unsafe_minz])
            safe_interval_final.append([grasp_unsafe_maxz, safe_interval_max])

    return safe_interval_final


def getSafeGraspPos(container_id, recording_id, replayFrameID, pred_data, safe_grasp_debug_lines, lefthand_trajectory, righthand_trajectory, start_frame, container_uid):
    traj = pred_data['container_trajectory']
    max_rad = pred_data['container_width'][0]/2.0
    cont_minz = pred_data['min_z_container_offset']
    cont_maxz = pred_data['max_z_container_offset']
    prev_debug_lines_hori = safe_grasp_debug_lines[0]
    prev_debug_lines_vert = safe_grasp_debug_lines[1]

    # OpenPose hand pose file
    lefthand_keypoints = lefthand_trajectory[replayFrameID]
    righthand_keypoints = righthand_trajectory[replayFrameID]

    left_hand_dist_to_cont = np.linalg.norm(lefthand_keypoints[9] - np.asarray(traj[replayFrameID]))
    right_hand_dist_to_cont = np.linalg.norm(righthand_keypoints[9] - np.asarray(traj[replayFrameID]))
    if right_hand_dist_to_cont > left_hand_dist_to_cont:
        grasphand = 'L'
    else:
        grasphand = 'R'

    # Find left hand orientation
    lefthand_dirv_a = lefthand_keypoints[9] - lefthand_keypoints[0]
    lefthand_dirv_b = lefthand_keypoints[13] - lefthand_keypoints[0]
    lefthand_dirv = lefthand_dirv_a + lefthand_dirv_b
    lefthand_dirv = lefthand_dirv / np.linalg.norm(lefthand_dirv)
    lefthand_w_temp = lefthand_keypoints[5] - lefthand_keypoints[9]
    lefthand_w_temp = lefthand_w_temp / np.linalg.norm(lefthand_w_temp)
    lefthand_up = np.cross(lefthand_w_temp, lefthand_dirv)
    lefthand_up = lefthand_up / np.linalg.norm(lefthand_up)
    lefthand_w = -np.cross(lefthand_up, lefthand_dirv)
    lefthand_w = lefthand_w / np.linalg.norm(lefthand_w)

    # Find right hand orientation
    righthand_dirv_a = righthand_keypoints[9] - righthand_keypoints[0]
    righthand_dirv_b = righthand_keypoints[13] - righthand_keypoints[0]
    righthand_dirv = righthand_dirv_a + righthand_dirv_b
    righthand_dirv = righthand_dirv / np.linalg.norm(righthand_dirv)
    righthand_w_temp = righthand_keypoints[5] - righthand_keypoints[9]
    righthand_w_temp = righthand_w_temp / np.linalg.norm(righthand_w_temp)
    righthand_up = np.cross(righthand_w_temp, righthand_dirv)
    righthand_up = righthand_up / np.linalg.norm(righthand_up)
    righthand_w = -np.cross(righthand_up, righthand_dirv)
    righthand_w = righthand_w / np.linalg.norm(righthand_w)

    # find safe grasp area
    if grasphand == 'L':
        up_v = lefthand_up
        dir_v = lefthand_dirv
        w_v = lefthand_w
    else:
        up_v = righthand_up
        dir_v = righthand_dirv
        w_v = righthand_w

    grasp_pos = {}
    grasp_pos['target_z'] = 0.0
    grasp_pos['max_z'] = -1.0
    grasp_pos['min_z'] = -1.0
    grasp_orn = {}
    grasp_orn['dir_v'] = dir_v
    grasp_orn['up_v'] = up_v
    grasp_orn['w_v'] = w_v

    gripper_safety = 0.011
    finger_safety = 0.01
    safety_offset = finger_safety + gripper_safety

    # find unsafe area
    left_grasp_unsafe_maxz, left_grasp_unsafe_minz = getUnsafeGrasp(lefthand_keypoints, traj, max_rad, safety_offset, replayFrameID, up_v, dir_v, w_v)
    right_grasp_unsafe_maxz, right_grasp_unsafe_minz = getUnsafeGrasp(righthand_keypoints, traj, max_rad, safety_offset, replayFrameID, up_v, dir_v, w_v)

    # find graspable area
    safe_interval = [[cont_minz + (gripper_safety/2.0), cont_maxz - (gripper_safety/2.0)]]
    # take difference between safe area and unsafe area
    safe_interval = getSafeUnsafeDiff(left_grasp_unsafe_maxz, left_grasp_unsafe_minz, safe_interval)
    safe_interval = getSafeUnsafeDiff(right_grasp_unsafe_maxz, right_grasp_unsafe_minz, safe_interval)
    if len(safe_interval) == 0:
        for debug_line in prev_debug_lines_hori:
            p.removeUserDebugItem(debug_line)
        for debug_line in prev_debug_lines_vert:
            p.removeUserDebugItem(debug_line)
        return False, grasp_pos, grasp_orn, [[], []]
    safe_interval_abs = []
    for si in safe_interval:
        safe_interval_abs.append(max(si) - min(si))

    safe_interval = safe_interval[safe_interval_abs.index(max(safe_interval_abs))]
    grasp_maxz = max(safe_interval)
    grasp_minz = min(safe_interval)
    grasp_z = (grasp_maxz + grasp_minz) / 2.0

    # if replayFrameID >= start_frame:
    if True:

        # Normalise relative to object centroid
        grasp_area_bottom_left  = traj[replayFrameID] - max_rad * up_v + max_rad * dir_v + grasp_minz * w_v

        # project onto container using raycast
        grasp_area_res = 20
        ray_test_start = []
        ray_test_end = []
        for height in np.linspace(0, grasp_maxz-grasp_minz, grasp_area_res):
            ray_test_start_temp = []
            ray_test_end_temp = []
            for width in np.linspace(0.0, max_rad*2.0, grasp_area_res):
                ray_test_start_temp.append(grasp_area_bottom_left + width * up_v + height * w_v)
                ray_test_end_temp.append(grasp_area_bottom_left + width * up_v - 0.5 * dir_v + height * w_v)
            ray_test_start.append(ray_test_start_temp)
            ray_test_end.append(ray_test_end_temp)

        grasp_area_display_line = []
        for line_id in range(len(ray_test_start)):
            ray_test_results = p.rayTestBatch(ray_test_start[line_id], ray_test_end[line_id])
            grasp_area_display_line_temp = []
            for ray in ray_test_results:
                if ray[0] == container_uid:
                    grasp_area_display_line_temp.append(ray[3] + 0.02 * dir_v)
            if len(grasp_area_display_line_temp) > 0:
                grasp_area_display_line.append(grasp_area_display_line_temp)

        curr_debug_lines_hori = []
        curr_debug_lines_vert = []


        for j, line in enumerate(grasp_area_display_line):
            # horizontal lines
            for i, line_pos in enumerate(line):
                if i < len(line) - 1:
                    if len(prev_debug_lines_hori) > 0:
                        curr_debug_lines_hori.append(p.addUserDebugLine(line_pos, line[i + 1], (0, 0.7, 0), replaceItemUniqueId=prev_debug_lines_hori.pop(0)))
                    else:
                        curr_debug_lines_hori.append(p.addUserDebugLine(line_pos, line[i + 1], (0, 0.7, 0)))
            # vertical lines
            if j < len(grasp_area_display_line) - 1:
                if len(prev_debug_lines_vert) > 0:
                    curr_debug_lines_vert.append(p.addUserDebugLine(line[0], grasp_area_display_line[j + 1][0], (0, 0.7, 0), replaceItemUniqueId=prev_debug_lines_vert.pop(0)))
                    curr_debug_lines_vert.append(p.addUserDebugLine(line[-1], grasp_area_display_line[j + 1][-1], (0, 0.7, 0), replaceItemUniqueId=prev_debug_lines_vert.pop(0)))
                else:
                    curr_debug_lines_vert.append(p.addUserDebugLine(line[0], grasp_area_display_line[j + 1][0], (0, 0.7, 0)))
                    curr_debug_lines_vert.append(p.addUserDebugLine(line[-1], grasp_area_display_line[j + 1][-1], (0, 0.7, 0)))

        for debug_line in prev_debug_lines_hori:
            p.removeUserDebugItem(debug_line)
        for debug_line in prev_debug_lines_vert:
            p.removeUserDebugItem(debug_line)
        prev_debug_lines_hori = curr_debug_lines_hori
        prev_debug_lines_vert = curr_debug_lines_vert
    else:
        prev_debug_lines_hori = []
        prev_debug_lines_vert = []

    safe_grasp_debug_lines = [prev_debug_lines_hori, prev_debug_lines_vert]
    grasp_pos['target_z'] = grasp_z
    grasp_pos['max_z'] = grasp_maxz
    grasp_pos['min_z'] = grasp_minz

    return True, grasp_pos, grasp_orn, safe_grasp_debug_lines
