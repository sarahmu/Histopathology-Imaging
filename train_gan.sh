python train_gan.py -t /home/shared/data/center_4/patches_256/datasets/train/ \
-v /home/shared/data/center_4/patches_256/datasets/val/ \
-o /home/shared/results/gan_0530/ \
-d 2e-4 -g 2e-4 -b 0.5 -r 16000 -n 50 -l l2 \
-s 16 -e -a -c /gpu:0 -i 500