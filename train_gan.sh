python train_gan.py -t /home/shared/data/center_4/datasets/train/ \
-v /home/shared/data/center_4/datasets/val/ \
-o /home/shared/unet_results/0602_l/ \
-d 2e-4 -g 2e-4 -b 0.5 -r 1 -n 30 -l l1 \
-s 16 -e -a -c /gpu:0 -i 500