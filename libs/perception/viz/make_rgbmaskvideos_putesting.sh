#!/bin/bash

#DATAPATH=/homes/ax300/import/Datasets/CCM/10


#filenames=(0003 0010 0027 0031 0037 0044 0049 0052 0053 0060 0067 0071 0072 0076)


#for ff in ${filenames[*]}
#do
#  filename=${ff}_c1
#  echo $filename

#  python3 videoseg.py --videopath $DATAPATH --videoname $filename
#  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
#  rm $DATAPATH/rgbmask/tmp/*
#  
#  filename=${ff}_c2
#  echo $filename

#  python3 videoseg.py --videopath $DATAPATH --videoname $filename
#  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
#  rm $DATAPATH/rgbmask/tmp/*
#done

################################################################################
DATAPATH=/homes/ax300/import/Datasets/CCM/11


filenames=(0045 0047 0083 0028 0000 0039 0054 0012 0055 0046 0023 0005 0009 0079)


for ff in ${filenames[*]}
do
  filename=${ff}_c1
  echo $filename

  python3 videoseg.py --videopath $DATAPATH --videoname $filename
  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
  rm $DATAPATH/rgbmask/tmp/*
  
  filename=${ff}_c2
  echo $filename

  python3 videoseg.py --videopath $DATAPATH --videoname $filename
  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
  rm $DATAPATH/rgbmask/tmp/*
done

echo "Finished!"
