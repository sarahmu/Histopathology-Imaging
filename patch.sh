#!/bin/bash
source activate openslide
python patch.py -d /home/shared/data/center_3/imgs/ \
-c /home/shared/data/center_3/patches_256/color/ \
-g /home/shared/data/center_3/patches_256/gray/ \
-w 0.95 -t 0.1 -p 256 -l 625 -n 1 \
-o /home/shared/data/center_3/patches_256/log/
source deactivate openslide