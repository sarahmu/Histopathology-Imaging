import sys, getopt
from gan import resume_train_gan

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   'm:c:t:v:o:n:s:eai:d:', 
                                   ['meta_file=', 'checkpoint_path=', 'train_data_dir=', 'val_data_dir=', 'output_dir=', 
                                    'num_epochs=', 'batch_size=', 'eval_val', 'save_eval_img', 'num_eval_img=', 'device='])
    except getopt.GetoptError:
        print ('resume_train_gan.py -m <meta_file> -c <checkpoint_path> -t <train_data_dir> -v <val_data_dir> -o <output_dir> -n <num_epochs> -s <batch_size> -e -a -i <num_eval_img> -d <device>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-m','--meta_file'):
            meta_file = arg
        elif opt in ('-c', '--checkpoint_path'):
            checkpoint_path = arg
        elif opt in ('-t', '--train_data_dir'):
            train_data_dir = arg
        elif opt in ('-v', '--val_data_dir'):
            val_data_dir = arg
        elif opt in ('-o', '--output_dir'):
            output_dir = arg
        elif opt in ('-n', '--num_epochs'):
            num_epochs = int(arg)
        elif opt in ('-s', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-e', '--eval_val'):
            eval_val = True
        elif opt in ('-a', '--save_eval_img'):
            save_eval_img = True
        elif opt in ('-i', '--num_eval_img'):
            num_eval_img = int(arg)
        elif opt in ('-d', '--device'):
            device = arg

    print('Running resume_train_gan.py with parameters: --meta_file=' + meta_file + \
          ', --checkpoint_path=' + checkpoint_path + ', --train_data_dir=' + train_data_dir + \
          ', --val_data_dir=' + val_data_dir + ', output_dir=' + output_dir + \
          ', --num_epochs=' + str(num_epochs) + ', --batch_size=' + str(batch_size) + ', --eval_val=' + str(eval_val) + \
          ', --save_eval_img=' + str(save_eval_img) + ', --num_eval_img=' + str(num_eval_img) + ', --device=' + device)

    resume_train_gan(meta_file=meta_file, checkpoint_path=checkpoint_path, train_data_dir=train_data_dir, 
                     val_data_dir=val_data_dir, output_dir=output_dir, num_epochs=num_epochs, batch_size=batch_size, 
                     eval_val=eval_val, save_eval_img=save_eval_img, num_eval_img=num_eval_img, device=device, 
                     img_dim=256)
    print('Finished resume training GAN!')
