import numpy as np
import tensorflow as tf

def encoder(gray_imgs, latent_dim, conv_filters=[32,16,8], kernel_sizes=[3,3,3], conv_strides=[2,2,1], 
            act=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer(), data_type=tf.float32):
    """
    Encode the input images into latent representations

    Inputs:
    - gray_imgs: Tensor for a batch of gray images with shape B x W x H x C
    - latent_dim: Integer for the latent dimension
    - conv_filers: List of integers for the numbers of filers
    - kernel_sizes: List of integers for the convolution kernel sizes
    - conv_strides: List of integers for the stride sizes
    - act: the activation function (non-linearity) for all the layers
    - initializer: the weight initializer for all the layers
    - data_type: Tensor data type for random Gaussian sampling

    Outputs:
    - z: Tensor for the latent representations with shape B x latent_dim
    - latent_mean: Tensor for the encoder latent means with shape B x latent_dim. Useful for KL divergence.
    - latent_sd: Tensor for the encoder latent standard devations withe shape B x latent_dim. Useful for KL divergence.
    """
    B = tf.shape(gray_imgs)[0] # batch size
    x = tf.layers.conv2d(inputs=gray_imgs, filters=conv_filters[0], kernel_size=kernel_sizes[0], 
                         strides=conv_strides[0], padding='same', activation=act, kernel_initializer=initializer)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(inputs=x, filters=conv_filters[1], kernel_size=kernel_sizes[1], 
                         strides=conv_strides[1], padding='same', activation=act, kernel_initializer=initializer)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(inputs=x, filters=conv_filters[2], kernel_size=kernel_sizes[2], 
                         strides=conv_strides[2], padding='same', activation=act, kernel_initializer=initializer)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.contrib.layers.flatten(x)
    latent_mean = tf.layers.dense(inputs=x, units=latent_dim, activation=act, kernel_initializer=initializer)
    latent_sd = tf.layers.dense(inputs=x, units=latent_dim, activation=act, kernel_initializer=initializer)

    # Apply the reparameterization trick to generate latent samples
    epsilon = tf.random_normal(shape=[B, latent_dim], dtype=data_type)
    latent_samples = latent_mean + tf.multiply(epsilon, latent_sd)
    return latent_samples, latent_mean, latent_sd

def test_encoder(device='/gpu:0'):
    """Unit test to for the encoder method. Check for output dimensions"""
    tf.reset_default_graph()
    B, H, W, C = 64, 256, 256, 1
    latent_dim = 16
    with tf.device(device):
        gray_imgs = tf.zeros((B, H, W, C))
        latent_samples, latent_mean, latent_sd = encoder(gray_imgs, latent_dim)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latent_samples_np, latent_mean_np, latent_sd_np = sess.run([latent_samples, latent_mean, latent_sd])
        print('Output shape should be (%d, %d)' % (B, latent_dim))
        print('latent_samples shape: ' + str(latent_samples_np.shape))
        print('latent_mean shape: ' + str(latent_mean_np.shape))
        print('latent_sd shape: ' + str(latent_sd_np.shape))

def decoder(latent_samples, input_dim, output_dim=500, input_channels=1, output_channels=4, deconv_filters=[4,8,16], 
            kernel_sizes=[3,3,3], deconv_strides=[1,1,1], act=tf.nn.relu, 
            initializer=tf.contrib.layers.xavier_initializer()):
    """
    Decode the latent samples into output color images

    Inputs:
    - latent_samples: Tensor for the samples drawn from the latent distribution with shape B x latent_dim
    - input_dim: Integer for the width and height of the first transformation of latent_samples
    - output_dim: Integer for the width and height of the final output
    - input_channels: Integer for the number of channels of the first transformation of latent_samples
    - output_channels: Integer for the number of channels of the final output
    - deconv_filters: List of integers for the deconvolution numbers of filters
    - kernel_sizes: List of integers for the deconvolution kernel sizes
    - deconv_strides: List of integers for the deconvolution strides
    - act: the activation function (non-linearity) for all the layers
    - initializer: the weight initializer for all the layers

    Outputs:
    - color_imgs: Tensor for a batch of output color images with shape B x output_dim x output_dim x output_channels
    """
    x = tf.layers.dense(inputs=latent_samples, units=input_dim*input_dim, activation=act, 
                        kernel_initializer=initializer)
    x = tf.reshape(x, [-1, input_dim, input_dim, input_channels])
    x = tf.layers.conv2d_transpose(inputs=x, filters=deconv_filters[0], kernel_size=kernel_sizes[0], 
                                   strides=deconv_strides[0], padding='same', activation=act, 
                                   kernel_initializer=initializer)
    x = tf.layers.conv2d_transpose(inputs=x, filters=deconv_filters[1], kernel_size=kernel_sizes[1], 
                                   strides=deconv_strides[1], padding='same', activation=act, 
                                   kernel_initializer=initializer)
    x = tf.layers.conv2d_transpose(inputs=x, filters=deconv_filters[2], kernel_size=kernel_sizes[2], 
                                   strides=deconv_strides[2], padding='same', activation=act, 
                                   kernel_initializer=initializer)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=output_dim*output_dim*output_channels, activation=act, 
                        kernel_initializer=initializer)
    color_imgs = tf.reshape(x, [-1, output_dim, output_dim, output_channels])
    return color_imgs

def test_decoder(device='/gpu:0'):
    """Unit test to for the decoder method. Check for output dimensions"""
    tf.reset_default_graph()
    B = 64
    latent_dim = 8
    input_dim, output_dim, input_channels, output_channels = 4, 500, 1, 4
    with tf.device(device):
        latent_samples = tf.zeros((B, latent_dim))
        color_imgs = decoder(latent_samples, input_dim, output_dim, input_channels, output_channels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        color_imgs_np = sess.run(color_imgs)
        print('Output shape should be (%d, %d, %d, %d)' % (B, output_dim, output_dim, output_channels))
        print('color_imgs shape: ' + str(color_imgs_np.shape))