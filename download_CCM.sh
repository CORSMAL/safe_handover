#!/bin/bash
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


# Download zip files of the RGB recordings
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view1_rgb.z01 -P data/CCM/
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view1_rgb.zip -P data/CCM/
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view2_rgb.z01 -P data/CCM/
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view2_rgb.zip -P data/CCM/

# Download zip files of the calibration parameters
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view1_calib.zip -P data/CCM/
wget -v http://corsmal.eecs.qmul.ac.uk/data/CCM/train/ccm_train_view2_calib.zip -P data/CCM/

# Join multi-part zip files
zip -FF data/CCM/ccm_train_view1_rgb.zip --out data/CCM/ccm_train_view1_rgb_full.zip
zip -FF data/CCM/ccm_train_view2_rgb.zip --out data/CCM/ccm_train_view2_rgb_full.zip

# Unzip then remove the downloaded zip files
unzip data/CCM/ccm_train_view1_rgb_full.zip -d data/CCM/
unzip data/CCM/ccm_train_view2_rgb_full.zip -d data/CCM/
unzip data/CCM/ccm_train_view1_calib.zip -d data/CCM/
unzip data/CCM/ccm_train_view2_calib.zip -d data/CCM/

rm data/CCM/ccm_train_view*
rm data/CCM/ccm_train_view*