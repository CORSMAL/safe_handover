#!/bin/bash

OBJID=6

#filenames=(fi0_fu0_b0_l0 fi1_fu1_b0_l0 fi1_fu2_b0_l0 fi2_fu1_b0_l0 fi2_fu2_b0_l0 fi3_fu1_b0_l0 fi3_fu2_b0_l0)

filenames=(fi0_fu0_b0_l0)

FPS=5

#for OBJID in 1 2 3 4 5 6
#do

  DATAPATH=/homes/ax300/import/Datasets/CCM/$OBJID
  RESPATH=../../results/$OBJID

  for ss in 0
  do
    for ff in ${filenames[*]}
    do
      filename=s${ss}_${ff}
      echo $filename

      python3 stereo_videoseg.py --videopath $DATAPATH --videoname $filename --respath $RESPATH
      ffmpeg -framerate $FPS -i $RESPATH/rgbmask/tmp/%04d.png -c:v libx264 -pix_fmt yuv420p  $RESPATH/rgbmask/$filename.mp4
      rm $RESPATH/rgbmask/tmp/*

    done
  done
#done

echo "Finished!"
