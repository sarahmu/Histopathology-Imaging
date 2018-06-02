# Center 4 test set
python evaluate_trained_gan.py -m '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19.meta' \
-c '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19' \
-e /home/shared/data/center_4/patches_256/datasets/test/ \
-o /home/shared/results/gan_0531/test_results/ \
-i 10000

# Center 0 
python evaluate_trained_gan.py -m '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19.meta' \
-c '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19' \
-e /home/shared/data/center_0/patches_256/ \
-o /home/shared/results/gan_0531/center_0_results/ \
-i 2500

# Center 1
python evaluate_trained_gan.py -m '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19.meta' \
-c '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19' \
-e /home/shared/data/center_1/patches_256/ \
-o /home/shared/results/gan_0531/center_1_results/ \
-i 2500

# Center 2
python evaluate_trained_gan.py -m '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19.meta' \
-c '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19' \
-e /home/shared/data/center_2/patches_256/ \
-o /home/shared/results/gan_0531/center_2_results/ \
-i 2500

# Center 3
python evaluate_trained_gan.py -m '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19.meta' \
-c '/home/shared/results/gan_0531/resumed/trained_sess/Dlr=0.0002_Glr=0.0002_beta1=0.5_reg=16000.0_loss=l1_epoch_19' \
-e /home/shared/data/center_3/patches_256/ \
-o /home/shared/results/gan_0531/center_3_results/ \
-i 2500