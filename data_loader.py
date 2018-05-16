import numpy as np
from PIL import Image
import os
import random

class Dataset(object):
    def __init__(self, gray_dir, color_dir, batch_size, img_dim, color_channels=4, shuffle=False, data_type='float32'):
        """
        Construct a Dataset object to iterate over grayscale images and their corresponding color images, if any

        Inputs:
        - gray_dir: String for the directory storing the gray images, with the ending slash
        - color_dir: String for the directory storing the color images, with the ending slahs. 
        None if there is no groundtruth color images (at testing time)
        - batch_size: Integer giving number of images per minibatch
        - img_dim: Integer for the width and height of the images
        - color_channels: Integer for the number of channels in the color images
        - shuffle: Boolean for shuffling the data on each epoch
        - data_type: String for specifying the numpy data type to convert to for the images
        """
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.color_channels = color_channels
        self.shuffle = shuffle
        self.data_type = data_type
        self.gray_img_names = os.listdir(gray_dir)
        if color_dir is None:
            self.color_img_names = None
        else:
            self.color_img_names = os.listdir(color_dir)

    # Convert a single image to numpy array with shape 1 x H x W x C
    def img_to_np(self, filename, img_type):
        assert img_type == 'gray' or img_type == 'color', "img_type has to be 'gray' or 'color'"
        img = Image.open(filename)
        data = np.array(img, dtype=self.data_type)
        if img_type == 'gray':
            data = data.reshape((1, *data.shape, 1))
        elif img_type == 'color':
            data = data.reshape((1, *data.shape))
        return data

    # Given a list of gray image files, create a numpy array with shape B x H x W x C for the gray images and
    # the color images, where N is the batch size
    def imgs_to_batch(self, gray_filelist, color_filelist=None):
        B = len(gray_filelist)
        gray_batch = np.zeros((B, self.img_dim, self.img_dim, 1))
        for i in range(B):
            gray_batch[i,:,:,:] = self.img_to_np(gray_filelist[i], 'gray')
        if color_filelist is not None:
            color_batch = np.zeros((B, self.img_dim, self.img_dim, self.color_channels))
            for i in range(B):
                color_batch[i,:,:,:] = self.img_to_np(color_filelist[i], 'color')
        else:
            color_batch = None
        return gray_batch, color_batch

    # Iterative method for generating a batch of gray image data and color image data
    def __iter__(self):
        N = len(self.gray_img_names)
        B = self.batch_size
        if self.shuffle:
            random.shuffle(self.gray_img_names)
        color_img_names = [img_name[5:] for img_name in self.gray_img_names] # remove the prefix 'gray_'
        gray_files = [self.gray_dir + img_name for img_name in self.gray_img_names]
        if self.color_dir is not None:
            color_files = [self.color_dir + img_name for img_name in color_img_names]
            return iter(self.imgs_to_batch(gray_files[i:i+B], color_files[i:i+B]) for i in range(0, N, B))
        else:
            return iter(self.imgs_to_batch(gray_files[i:i+B], None) for i in range(0, N, B))