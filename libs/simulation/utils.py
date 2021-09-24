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

from scipy.spatial.transform import Rotation as Rot
import cv2
import numpy as np

TRAJECTORY_Z_OFFSET = 0.625

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def mat2quaternion(R):
    r = Rot.from_matrix(R)
    return r.as_quat()


def quaternion2mat(quat):
    r = Rot.from_quat(quat)
    return r.as_matrix()


def vec2quaternion(dir_x, dir_y, dir_z):

    target_frame = np.asarray([dir_x, dir_y, dir_z])
    target_frame = np.transpose(target_frame)

    orignal_frame = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    R = np.matmul(target_frame, np.linalg.inv(orignal_frame))

    return mat2quaternion(R)

def kalmanFilter(measured_traj):
    sp = 0.05
    dt = 1. / 30.
    v = dt
    a = 0.5 * (dt ** 2)

    kalman = cv2.KalmanFilter(9, 3, 0)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, v, 0, 0, a, 0, 0],
        [0, 1, 0, 0, v, 0, 0, a, 0],
        [0, 0, 1, 0, 0, v, 0, 0, a]
    ], np.float32)

    kalman.transitionMatrix = np.array([
        [1, 0, 0, v, 0, 0, a, 0, 0],
        [0, 1, 0, 0, v, 0, 0, a, 0],
        [0, 0, 1, 0, 0, v, 0, 0, a],
        [0, 0, 0, 1, 0, 0, v, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, v, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, v],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)

    kalman.processNoiseCov = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32) * 0.007

    kalman.measurementNoiseCov = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], np.float32) * sp

    kalman.statePre = np.array([
        [np.float32(measured_traj[0][0])], [np.float32(measured_traj[0][1])], [np.float32(measured_traj[0][2])]
        , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
        , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
    ])

    mp = np.array((3, 1), np.float32)  # measurement
    tp = np.zeros((3, 1), np.float32)
    last_safe = 0
    filtered_traj = []
    for i in range(len(measured_traj)):

        if -0.05 < measured_traj[i][0] < 0.7 and -0.5 < measured_traj[i][1] < 1.0 and 0.5 < measured_traj[i][2] < 1.5:
            last_safe = i
        else:
            pass

        mp = np.array([
            [np.float32(measured_traj[last_safe][0])],
            [np.float32(measured_traj[last_safe][1])],
            [np.float32(measured_traj[last_safe][2])]
        ])

        kalman.correct(mp)
        tp = kalman.predict()
        filtered_traj.append([float(tp[0]), float(tp[1]), float(tp[2])])

    return np.asarray(filtered_traj)