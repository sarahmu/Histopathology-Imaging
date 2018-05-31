python resume_train_gan.py -m '/home/shared/chris/Histopathology-Imaging/gan_testing/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l2_epoch_29.meta' \
-c '/home/shared/chris/Histopathology-Imaging/gan_testing/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l2_epoch_29' \
-t /home/shared/chris/Histopathology-Imaging/testing/ \
-v /home/shared/chris/Histopathology-Imaging/testing/ \
-o /home/shared/chris/Histopathology-Imaging/gan_testing/test_resume_train/ \
-n 40 -s 16 -e -a -i 25 -d /gpu:0