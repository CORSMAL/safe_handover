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

# Download zip files of the recordings
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/1_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/2_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/3_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/4_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/5_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/Training/6_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/PuTesting/10_rgb.zip -P data/videos/
wget -v http://corsmal.eecs.qmul.ac.uk/data/ICME20/PuTesting/11_rgb.zip -P data/videos/


# Unzip then remove the downloaded zip files
unzip data/videos/1_rgb.zip -d data/videos/
rm  data/videos/1_rgb.zip
rm  data/videos/License.txt
unzip data/videos/2_rgb.zip -d data/videos/
rm  data/videos/2_rgb.zip
rm  data/videos/License.txt
unzip data/videos/3_rgb.zip -d data/videos/
rm  data/videos/3_rgb.zip
rm  data/videos/License.txt
unzip data/videos/4_rgb.zip -d data/videos/
rm  data/videos/4_rgb.zip
rm  data/videos/License.txt
unzip data/videos/5_rgb.zip -d data/videos/
rm  data/videos/5_rgb.zip
rm  data/videos/License.txt
unzip data/videos/6_rgb.zip -d data/videos/
rm  data/videos/6_rgb.zip
rm  data/videos/License.txt
unzip data/videos/10_rgb.zip -d data/videos/
tar -xvzf data/videos/10_rgb.tar.gz -C data/videos/
rm  data/videos/10_rgb.tar.gz
rm  data/videos/10_rgb.zip
rm  data/videos/License.txt
unzip data/videos/11_rgb.zip -d data/videos/
tar -xvzf data/videos/11_rgb.tar.gz -C data/videos/
rm  data/videos/11_rgb.tar.gz
rm  data/videos/11_rgb.zip