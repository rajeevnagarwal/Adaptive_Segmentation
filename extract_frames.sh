#!bin/bash

ffmpeg -i ./test_images/video.mkv  -vf fps=1 ./test_images/thumb%d.png
shopt -s extglob
rm ./test_images/!(*.png)
