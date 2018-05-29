from os import listdir
from shutil import copyfile
def copy_file(org_path_color, org_path_gray, dest_path_color, dest_path_gray, num_files = 100):
    files = [f for f in listdir(org_path_color)]
    for i in range(num_files):
        color_file = files[i]
        gray_file = 'gray_' + color_file
        print('Copying ' + color_file)
        copyfile(org_path_color + color_file, dest_path_color + color_file)
        print('Copying ' + gray_file)
        copyfile(org_path_gray + gray_file, dest_path_gray + gray_file)