import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os.path
from data_loader import Dataset
from utils import local_clock, preprocess, postprocess, show_images, save_image
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from scipy.stats import pearsonr as compare_pearson

def compare_hist_intersection(img1, img2):
    """
    Calculate the histogram intersection between two images to compare color distgributions.

    Inputs:
    - img1: Numpy array for an image of shape (H, W, C)
    - img2: Numpy array for an image of shape (H, W, C)

    Outputs: Value of histogram intersection [0, 1]
    """
    H, W, C = img1.shape
    channel_intersections = []
    for i in range(C):
        hist1, bins = np.histogram(img1[:,:,i], bins=256, range=(0, 255), density=True)
        hist2, bins = np.histogram(img2[:,:,i], bins=256, range=(0, 256), density=True)
        bins = np.diff(bins)
        total = 0
        for j in range(len(bins)):
            total += min(bins[j] * hist1[j], bins[j] * hist2[j])
        channel_intersections.append(total)
    return np.average(channel_intersections)

def evaluate_model(sess, graph_gray, graph_color, graph_training, graph_D_loss, graph_G_loss, graph_img_loss, graph_G_sample, 
    dataset, log_filename, log_note, csv_filename, output_imgs=False, img_dir=None):
    """
    Evaluate the GAN model performance on a given dataset.
    
    Inputs:
    - sess: TensorFlow session with the trained model and variables
    - graph_gray: TensorFlow placeholder for the gray images
    - graph_color: TensorFlow placeholder for the color images
    - graph_training: TensorFlow placeholder for is_training
    - graph_D_loss: TensorFlow node for D_loss
    - graph_G_loss: TensorFlow node for G_loss
    - graph_img_loss: TensorFlow node for img_loss
    - graph_G_sample: TensorFlow node for G_sample
    - dataset: Dataset object for the data set to evaluate on. The attribute shuffle should be False.
    - log_filename: String for the file name of the performance log file 
    - log_note: String for the note for the log file
    - csv_filename: String for the csv file name of the index and metric for each image in the dataset
    - output_imgs: Boolean for whether to save generated images
    - img_dir: String for the output directory of the images
    
    Outputs:
    None. Save the mean and standard deviation of (i) SSIM, (ii) PSNR, (iii) Pearson Correlation, (iv) Histogram Intersection
    , and (v) Euclidean norm distance to a log file. Save the average discriminator and generator loesses to the same log file. 
    Save a csv file with the original file name, index for the generated image, and the metrics. 
    If true, save the generated images.
    """
    assert dataset.shuffle == False, "shuffle should be turned off for the dataset"
    D_loss_list = []
    G_loss_list = []
    img_loss_list = []
    ssim_list = []
    psnr_list = []
    pearson_list = []
    hist_intersection_list = []
    mse_list = []
    gray_img_names = dataset.gray_img_names
    csv_pd = pd.DataFrame(index=pd.RangeIndex(0, len(gray_img_names), 1), 
                          columns=['filename', 'ssim', 'psnr', 'pearson', 'hist_intersection', 'mse'])
    counter = 0
    for t, (gray_img_np, color_img_np) in enumerate(dataset):
        gray_processed_np = preprocess(gray_img_np)
        color_processed_np = preprocess(color_img_np)
        feed_dict = {graph_gray: gray_processed_np, graph_color: color_processed_np, graph_training: False}
        D_loss_np = sess.run(graph_D_loss, feed_dict=feed_dict)
        G_loss_np, img_loss_np = sess.run([graph_G_loss, graph_img_loss], feed_dict=feed_dict)

        D_loss_list.append(D_loss_np)
        G_loss_list.append(G_loss_np)
        img_loss_list.append(img_loss_np)

        samples_np = sess.run(graph_G_sample, feed_dict=feed_dict)
        samples_np = postprocess(samples_np)
        N, H, W, C = samples_np.shape
        for i in range(N):
            original_imgname = gray_img_names[counter]
            sample_imgname = 'gen' + original_imgname[4:]
            sample_img = samples_np[i,:,:,:] # sample image that has been postprocessed

            # Save the generated image
            if output_imgs and img_dir != None:
                save_image(sample_img, img_dir + sample_imgname, post_process=False)

            true_img = color_img_np[i,:,:,:] # color image that has not been preprocessed

            # Convert the images to uint8
            sample_img = sample_img.astype(np.uint8)
            true_img = true_img.astype(np.uint8)

            # Calculate the metrics
            ssim = compare_ssim(sample_img, true_img, multichannel=True)
            psnr = compare_psnr(sample_img, true_img, data_range=255)
            pearson = compare_pearson(sample_img.flatten(), true_img.flatten())[0]
            hist_intersection = compare_hist_intersection(sample_img, true_img)
            mse = compare_mse(sample_img, true_img)

            # Append the metrics to their lists for mean and standard deviation
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            pearson_list.append(pearson)
            hist_intersection_list.append(hist_intersection)
            mse_list.append(mse)

            # Append the metrics to the data frame for the output csv file
            csv_pd.at[counter, 'filename'] = sample_imgname
            csv_pd.at[counter, 'ssim'] = ssim
            csv_pd.at[counter, 'psnr'] = psnr
            csv_pd.at[counter, 'pearson'] = pearson
            csv_pd.at[counter, 'hist_intersection'] = hist_intersection
            csv_pd.at[counter, 'mse'] = mse
            counter += 1

    # Calculate the means and standard deviations
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)

    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)

    pearson_mean = np.mean(pearson_list)
    pearson_std = np.std(pearson_list)

    hist_intersection_mean = np.mean(hist_intersection_list)
    hist_intersection_std = np.std(hist_intersection_list)

    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)

    D_loss_mean = np.mean(D_loss_list)
    G_loss_mean = np.mean(G_loss_list)
    img_loss_mean = np.mean(img_loss_list)

    # Save the results to the log file
    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as log_handle:
            log_handle.write(local_clock() + '  ' + log_note + '\n')
    else:
        with open(log_filename, 'a') as log_handle:
            log_handle.write(local_clock() + '  ' + log_note + '\n')
    with open(log_filename, 'a') as log_handle:
        log_handle.write('D loss: %0.4f\n' % (D_loss_mean))
        log_handle.write('G loss: %0.4f\n' % (G_loss_mean))
        log_handle.write('img loss: %0.4f\n' % (img_loss_mean))

        log_handle.write('SSIM mean: %0.4f  std: %0.4f\n' % (ssim_mean, ssim_std))
        log_handle.write('PSNR mean: %0.4f  std: %0.4f\n' % (psnr_mean, psnr_std))
        log_handle.write('Pearson mean: %0.4f  stf: %0.4f\n' % (pearson_mean, pearson_std))
        log_handle.write('Histogram Intersection mean: %0.4f  std: %0.4f\n' % (hist_intersection_mean, hist_intersection_std))
        log_handle.write('MSE mean: %0.4f  std: %0.4f\n' % (mse_mean, mse_std))
        log_handle.write('\n')

    # Save the results to a csv file
    csv_pd.to_csv(csv_filename, index=False)