#!bin/bash

./youtube-dl.exe $1 -o video
mv video.* ./test_images/.
scp -r ./test_images team5@192.168.28.5:DL/Project/.

