import os
import shutil
import random
import sys, getopt

def transfer_imgs(color_samples, start_idx, end_idx, source_dirs, dest_dirs, dataset, method):
    """
    Transfer color and gray images from source directories to destination directories
    
    Inputs:
    - color_samples: List of the image file names of the color images
    - start_idx: Integer for indexing the first file in color_samples to transfer
    - end_idx: Integer for indexing the last file in color_samples to transfer
    - source_dirs: Dictionary with keys 'color' and 'gray' to specify the source directories
    - dest_dirs: Dictionary with keys 'train', 'train_debug', 'val', 'test' for dictionaries with keys 'color'
      and 'gray' for the destination directories for each dataset
    - dataset: String for the dataset to create
    - method: shutil.copyfile for copying or shutil.move for moving the files
    
    Outputs: None
    """
    for i in range(start_idx, end_idx):
        color_file = color_samples[i]
        gray_file = 'gray_' + color_file
        method(source_dirs['color'] + color_file, dest_dirs[dataset]['color'] + color_file)
        method(source_dirs['gray'] + gray_file, dest_dirs[dataset]['gray'] + gray_file)
    return

def create_dest_dirs(master_dest_dir, datasets=['train', 'train_debug', 'val', 'test']):
    """
    Create a dictionary of the destination directories. Also create the destination directories if not existent.
    
    Inputs:
    - master_dest_dir: String for the master directory for all the destination directories
    - datasets: List of strings for the datasets to create
    
    Outputs: None
    """
    dest_dirs = {}
    for dataset in datasets:
        dataset_master_dir = master_dest_dir + dataset + '/'
        color_dir = dataset_master_dir + 'color/'
        gray_dir = dataset_master_dir + 'gray/'
        
        dest_dirs[dataset] = {}
        dest_dirs[dataset]['color'] = color_dir
        dest_dirs[dataset]['gray'] = gray_dir

        if not os.path.exists(dataset_master_dir):
            os.makedirs(dataset_master_dir)
        if not os.path.exists(color_dir):
            os.makedirs(color_dir)
        if not os.path.exists(gray_dir):
            os.makedirs(gray_dir)
    return dest_dirs

def main(source_color_dir, source_gray_dir, master_dest_dir, train_size, train_debug_size, val_size, test_size):
    # Set up the directories
    source_dirs = {'color': source_color_dir, 'gray': source_gray_dir}
    dest_dirs = create_dest_dirs(master_dest_dir)
    
    # Find the overlapping color and gray source files in the source directories
    color_imgfiles = os.listdir(source_color_dir)
    gray_imgfiles = os.listdir(source_gray_dir)
    gray_sourcefiles = [file[5:] for file in gray_imgfiles]
    overlap = set.intersection(set(color_imgfiles), set(gray_sourcefiles))
    overlap = list(overlap)
    
    # Sample for the datasets
    total_size = train_size + val_size + test_size
    color_samples = random.sample(overlap, total_size)
    
    # Create the training debug set for overfitting
    transfer_imgs(color_samples, 0, train_debug_size, source_dirs, dest_dirs, 'train_debug', shutil.copyfile)
    
    # Create the training set
    transfer_imgs(color_samples, 0, train_size, source_dirs, dest_dirs, 'train', shutil.move)
    
    # Create the validation set
    transfer_imgs(color_samples, train_size, train_size + val_size, source_dirs, dest_dirs, 'val', shutil.move)
    
    # Create the testing set
    transfer_imgs(color_samples, train_size + val_size, total_size, source_dirs, dest_dirs, 'test', shutil.move)
    return

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   'c:g:m:t:d:v:s:', 
                                   ['source_color_dir=', 'source_gray_dir=', 'master_dest_dir=', 
                                    'train_size=', 'train_debug_size=', 'val_size=', 'test_size='])
    except getopt.GetoptError:
        print ('split_data.py -c <source_color_dir> -g <source_gray_dir> -m <master_dest_dir> -t <train_size> -d <train_debug_size> -v <val_size> -s <test_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-c','--source_color_dir'):
            source_color_dir = arg
        elif opt in ('-g', '--source_gray_dir'):
            source_gray_dir = arg
        elif opt in ('-m', '--master_dest_dir'):
            master_dest_dir = arg
        elif opt in ('-t', '--train_size'):
            train_size = int(arg)
        elif opt in ('-d', '--train_debug_size'):
            train_debug_size = int(arg)
        elif opt in ('-v', '--val_size'):
            val_size = int(arg)
        elif opt in ('-s', '--test_size'):
            test_size = int(arg)
    print('Running split_data.py with parameters: --source_color_dir=' + source_color_dir + \
          ', --source_gray_dir=' + source_gray_dir + ', --master_dest_dir=' + master_dest_dir + \
          ', --train_size=' + str(train_size) + ', --train_debug_size=' + str(train_debug_size) + \
          ', --val_size=' + str(val_size) + ', --test_size=' + str(test_size))
    main(source_color_dir, source_gray_dir, master_dest_dir, train_size, train_debug_size, val_size, test_size)
    print('Done splitting the datasets!')