#!/bin/bash
source activate openslide
python patch.py -d /home/shared/data/center_4/imgs/ -c /home/shared/data/center_4/patches/color/ -g /home/shared/data/center_4/patches/gray/ -w 0.95 -t 0.1 -p 224 -l 5000 -n 7 -o /home/shared/data/center_4/patches/log/
source deactivate openslide