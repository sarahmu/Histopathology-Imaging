import sys, getopt
from gan import evaluate_trained_gan

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   'm:c:e:o:i:', 
                                   ['meta_file=', 'checkpoint_path=', 'eval_data_dir=', 'output_dir=', 'num_eval_img='])
    except getopt.GetoptError:
        print ('evaluate_trained_gan.py -m <meta_file> -c <checkpoint_path> -e <eval_data_dir> -o <output_dir> -i <num_eval_img>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-m','--meta_file'):
            meta_file = arg
        elif opt in ('-c', '--checkpoint_path'):
            checkpoint_path = arg
        elif opt in ('-e', '--eval_data_dir'):
            eval_data_dir = arg
        elif opt in ('-o', '--output_dir'):
            output_dir = arg
        elif opt in ('-i', '--num_eval_img'):
            num_eval_img = int(arg)

    print('Running evaluate_trained_gan.py with parameters: --meta_file=' + meta_file + \
          ', --checkpoint_path=' + checkpoint_path + ', --eval_data_dir=' + eval_data_dir + \
          ', --output_dir=' + output_dir + ', --num_eval_img=' + str(num_eval_img))

    evaluate_trained_gan(meta_file=meta_file, checkpoint_path=checkpoint_path, eval_data_dir=eval_data_dir, 
                         output_dir=output_dir, num_eval_img=num_eval_img, batch_size=16, img_dim=256)
    print('Finished evaluating trained GAN!')