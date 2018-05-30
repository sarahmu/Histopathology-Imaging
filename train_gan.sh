python train_gan.py -t /home/shared/chris/Histopathology-Imaging/testing/ \
-v /home/shared/chris/Histopathology-Imaging/testing/ \
-o /home/shared/chris/Histopathology-Imaging/gan_testing/ \
-d 2e-4 -g 2e-4 -b 0.5 -r 16000 -n 30 -l l2 \
-s 16 -e -a -c /gpu:0 -i 50