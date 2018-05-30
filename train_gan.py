import sys, getopt
from gan import train_gan

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   't:v:o:d:g:b:r:n:l:s:eac:', 
                                   ['train_data_dir=', 'val_data_dir=', 'output_dir=', 
                                    'D_lr=', 'G_lr=', 'beta1=', 'reg=', 'num_epochs=', 'loss=', 'batch_size=', 
                                    'eval_val', 'save_eval_img', 'device='])
    except getopt.GetoptError:
        print ('train_gan.py -t <train_data_dir> -v <val_data_dir> -o <output_dir> -d <D_lr> -g <G_lr> -b <beta1> -r <reg> -n <num_epochs> -l <loss> -s <batch_size> -e -a -c <device>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-t','--train_data_dir'):
            train_data_dir = arg
        elif opt in ('-v', '--val_data_dir'):
            val_data_dir = arg
        elif opt in ('-o', '--output_dir'):
            master_dest_dir = arg
        elif opt in ('-d', '--D_lr'):
            D_lr = float(arg)
        elif opt in ('-g', '--G_lr'):
            G_lr = float(arg)
        elif opt in ('-b', '--beta1'):
            beta1 = float(arg)
        elif opt in ('-r', '--reg'):
            reg = float(arg)
        elif opt in ('-n', '--num_epochs'):
            num_epochs = int(arg)
        elif opt in ('-l', '--loss'):
            loss = arg
        elif opt in ('-s', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-e', '--eval_val'):
            eval_val = True
        elif opt in ('-a', '--save_eval_img'):
            save_eval_img = True
        elif opt in ('-c', '--device'):
            device = arg
    print('Running train_gan.py with parameters: --train_data_dir=' + train_data_dir + \
          ', --val_data_dir=' + val_data_dir + ', --output_dir=' + output_dir + \
          ', --D_lr=' + str(D_lr) + ', --G_lr=' + str(G_lr) + \
          ', --beta1=' + str(beta1) + ', --reg=' + str(reg) + ', --num_epochs=' + str(num_epochs) + \
          ', --loss=' + loss + ', --batch_size=' + str(batch_size) + ', --eval_val=' + str(eval_val) + \
          ', --save_eval_img=' + str(save_eval_img) + ', --device=' + device)

    train_gan(train_data_dir=train_data_dir, val_data_dir=val_data_dir, output_dir=output_dir, 
              D_lr=D_lr, G_lr=G_lr, beta1=beta1, reg=reg, num_epochs=num_epochs, loss=loss, 
              batch_size=batch_size, eval_val=eval_val, save_eval_img=save_eval_img, device=device)
