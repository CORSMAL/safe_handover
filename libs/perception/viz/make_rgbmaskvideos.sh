#!/bin/bash

DATAPATH=/homes/ax300/import/Datasets/CCM/1


#filenames=(fi0_fu0_b0_l0 fi1_fu1_b0_l0 fi1_fu2_b0_l0 fi2_fu1_b0_l0 fi2_fu2_b0_l0 fi3_fu1_b0_l0 fi3_fu2_b0_l0)

filenames=(fi0_fu0_b0_l0)

FPS=5

for ff in ${filenames[*]}
do
  filename=s0_${ff}_c1
  echo $filename

  python3 videoseg.py --videopath $DATAPATH --videoname $filename --cam_id 1
  ffmpeg -framerate $FPS -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
  rm $DATAPATH/rgbmask/tmp/*
  
  filename=s0_${ff}_c2
  echo $filename

  python3 videoseg.py --videopath $DATAPATH --videoname $filename --cam_id 2
  ffmpeg -framerate $FPS -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
  rm $DATAPATH/rgbmask/tmp/*
  
#  filename=s1_${ff}_c1
#  echo $filename

#  python3 videoseg.py --videopath $DATAPATH --videoname $filename
#  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
#  rm $DATAPATH/rgbmask/tmp/*
#  
#  filename=s1_${ff}_c2
#  echo $filename

#  python3 videoseg.py --videopath $DATAPATH --videoname $filename
#  ffmpeg -framerate 30 -i $DATAPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $DATAPATH/rgbmask/$filename.mp4
#  rm $DATAPATH/rgbmask/tmp/*
done
echo "Finished!"
