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

# Download zip files of the calibration data
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/1_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/2_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/3_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/4_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/5_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/6_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/PuTesting/10_others.zip -P data/calibration/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/PuTesting/11_others.zip -P data/calibration/


# Unzip then remove the downloaded zip files
unzip data/calibration/1_others.zip -d data/calibration/
rm  data/calibration/1_others.zip
rm  data/calibration/License.txt
unzip data/calibration/2_others.zip -d data/calibration/
rm  data/calibration/2_others.zip
rm  data/calibration/License.txt
unzip data/calibration/3_others.zip -d data/calibration/
rm  data/calibration/3_others.zip
rm  data/calibration/License.txt
unzip data/calibration/4_others.zip -d data/calibration/
rm  data/calibration/4_others.zip
rm  data/calibration/License.txt
unzip data/calibration/5_others.zip -d data/calibration/
rm  data/calibration/5_others.zip
rm  data/calibration/License.txt
unzip data/calibration/6_others.zip -d data/calibration/
rm  data/calibration/6_others.zip
rm  data/calibration/License.txt
unzip data/calibration/10_others.zip -d data/calibration/
tar -xvzf data/calibration/10_extra.tar.gz -C data/calibration/
rm  data/calibration/10_extra.tar.gz
rm  data/calibration/10_others.zip
rm  data/calibration/License.txt
unzip data/calibration/11_others.zip -d data/calibration/
tar -xvzf data/calibration/11_extra.tar.gz -C data/calibration/
rm  data/calibration/11_extra.tar.gz
rm  data/calibration/11_others.zip