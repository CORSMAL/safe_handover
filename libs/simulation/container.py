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

import math

import pybullet as p

from libs.simulation.utils import *

CONT_CONSTRAINT_FORCE = 100

class Container(object):

    def __init__(self, mesh_path, trajectory, mass, table_uids):
        self.mesh_path = mesh_path
        self.table_uids = table_uids

        self.trajectory = trajectory
        self.uid, self.cid = self.loadMesh()

        p.changeDynamics(self.uid, -1, mass)

    def loadMesh(self):
        init_pos = self.trajectory[0]

        # Spawn container
        uid = p.loadURDF(self.mesh_path, init_pos)
        # Create movement constraint
        cid = p.createConstraint(uid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], init_pos, p.getQuaternionFromEuler([0, -math.pi/2.0, 0]))

        # disable collision between container and table
        for col_objs in self.table_uids:
            p.setCollisionFilterPair(col_objs, uid, -1, -1, 0)

        return uid, cid

    def updatePose(self, replayFrameID, cont_quat):
        # get quat from hand
        # use own pos
        cont_pos = self.trajectory[replayFrameID + 1]
                
        p.changeConstraint(self.cid, cont_pos, cont_quat, maxForce=CONT_CONSTRAINT_FORCE)