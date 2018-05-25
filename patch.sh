#!/bin/bash
source activate openslide
python patch.py -d /home/shared/data/center_4/imgs/ -c /home/shared/data/center_4/patches_256/color/ -g /home/shared/data/center_4/patches_256/gray/ -w 0.95 -t 0.1 -p 256 -l 5000 -n 7 -o /home/shared/data/center_4/patches_256/log/
source deactivate openslide