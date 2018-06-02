import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os.path
import os
from data_loader import Dataset
from utils import local_clock, preprocess, postprocess, show_images, save_image
from evaluate import evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Generator
def generator(x, is_training, output_channels=4, filters = [64,64,128,256,512,512,512,512], kernel_size = 4, stride = 2):
    with tf.variable_scope('generator'):
        layers = []
        # Encoder:
        x = tf.layers.conv2d(inputs = x,
                             filters = filters[0],
                             kernel_size = 1,
                             strides = 1,
                             padding = 'same',
                             kernel_initializer = tf.contrib.layers.xavier_initializer()) 
        x = tf.layers.batch_normalization(x, training=is_training)
        x =  tf.nn.leaky_relu(x)
        layers.append(x)
        for i in range(1, len(filters)):
            x = tf.layers.conv2d(inputs = x,
                                 filters = filters[i],
                                 kernel_size = kernel_size,
                                 strides = stride,
                                 padding = 'same',
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())  
            x = tf.layers.batch_normalization(x, training=is_training)
            x =  tf.nn.leaky_relu(x)
        # save contracting path layers to be used for skip connections
            layers.append(x)
            
        
        # Decoder:
        for i in reversed(range(len(filters)-1)):
            x = tf.layers.conv2d_transpose(inputs = x,
                                           filters = filters[i],
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = 'same',
                                           kernel_initializer = tf.contrib.layers.xavier_initializer())
            x = tf.layers.batch_normalization(x, training=is_training)
            x =  tf.nn.relu(x)
        # concat the layer from the contracting path with the output of the current layer
        # concat only the channels (axis=3)
            x = tf.concat([layers[i], x], axis=3)
            # layers.append(x)
        x = tf.layers.conv2d(inputs = x,
                             filters = output_channels,
                             kernel_size = 1,
                             strides = 1,
                             padding = 'same',
                             activation = tf.nn.tanh,
                             kernel_initializer = tf.contrib.layers.xavier_initializer())   
        # layers.append(x)
        # return layers
        return x

# Discriminator
def discriminator(x, is_training, filters = [64,128,256,512] , kernel_size = 4, stride = 2): # conditional GAN
    """
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
                 Can be a single integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. 
             Can be a single integer to specify the same value for all spatial dimensions. 
             Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    
    filters: a series of 4x4 convolutional layers with stride 2 with the number of channels being doubled after each downsampling.
    All convolution layers are followed by batch normalization, leaky ReLU activation. 
    After the last layer, a convolution is applied to map to a 1 dimensional output, 
        followed by a sigmoid function to return a probability value of the input being real or fake
    """
    with tf.variable_scope("discriminator"): 
        # layers = []
        for i in range(len(filters)):
            x = tf.layers.conv2d(inputs = x,
                                 filters = filters[i],
                                 kernel_size = kernel_size,
                                 strides = stride,
                                 padding = 'same',
                                 kernel_initializer = tf.contrib.layers.xavier_initializer())           
            if i != 0: # Do not use batch-norm in the first layer
                x = tf.layers.batch_normalization(x, training=is_training)
            x =  tf.nn.leaky_relu(x)
            # layers.append(x)
        x = tf.contrib.layers.flatten(x)
        logit = tf.layers.dense(inputs = x, units=1, kernel_initializer = tf.contrib.layers.xavier_initializer())
        # layers.append(logit)
        # return layers
        return logit

def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Unnormalized score that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    
    Note: For the discriminator loss, do the averaging separately for
    its two components, and then add them together (instead of averaging once at the very end).
    """
    G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
    G_loss = tf.reduce_mean(G_loss)
    
    D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)
    D_real_loss = tf.reduce_mean(D_real_loss)
    D_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake)
    D_fake_loss = tf.reduce_mean(D_fake_loss)
    D_loss = D_real_loss + D_fake_loss
    return D_loss, G_loss

def l1_loss(fake_imgs, real_imgs, reg=128):
    """
    Compute the L1 loss between fake images and real images.
    
    Inputs:
    - fake_imgs: Tensor with shape [batch_size, H, W, C], output of generator
    - real_imgs: Tensor with shape [batch_size, H, W, C], fed into the graph
    - reg: Float for the regularization constant. Default to 128 for RGBA scheme (0-255).
    
    Outputs:
    - loss: L1 loss scalar
    """
    fake_flat = tf.contrib.layers.flatten(fake_imgs)
    real_flat = tf.contrib.layers.flatten(real_imgs)
    loss = tf.reduce_mean(tf.abs(fake_flat - real_flat))
    return reg * loss

def l2_loss(fake_imgs, real_imgs, reg=128):
    """
    Compute the L2 loss between fake images and real images.
    
    Inputs:
    - fake_imgs: Tensor with shape [batch_size, H, W, C], output of generator
    - real_imgs: Tensor with shape [batch_size, H, W, C], fed into the graph
    - reg: Float for the regularization constant. Default to 128 for RGBA scheme (0-255).
    
    Outputs:
    - loss: L1 loss scalar
    """
    fake_flat = tf.contrib.layers.flatten(fake_imgs)
    real_flat = tf.contrib.layers.flatten(real_imgs)
    loss = 2* tf.nn.l2_loss(fake_flat - real_flat)
    return reg * loss

def calculate_mse(fake_imgs, real_imgs, post_process=True):
    if post_process:
        fake_imgs = postprocess(fake_imgs)
        real_imgs = postprocess(real_imgs)
    fake_flat = tf.contrib.layers.flatten(fake_imgs)
    real_flat = tf.contrib.layers.flatten(real_imgs)
    return tf.losses.mean_squared_error(real_flat, fake_flat)

def get_solvers(D_lr=2e-4, G_lr=2e-4, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - D_lr: learning rate for the discriminator
    - G_lr: learning rate for the generator
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(D_lr, beta1)
    G_solver = tf.train.AdamOptimizer(G_lr, beta1)
    return D_solver, G_solver

def train_gan(train_data_dir, val_data_dir, output_dir, D_lr, G_lr, beta1, reg, num_epochs, 
              loss='l2', batch_size=16, eval_val=True, save_eval_img=True, num_eval_img=100, device='/gpu:0', img_dim=256):
    # Set up the image loss function
    if loss == 'l2':
        loss_method = l2_loss
    elif loss == 'l1':
        loss_method = l1_loss

    # Set up output directories    
    val_dir = output_dir + 'val_results/'
    val_img_dir = val_dir + 'imgs/'
    train_dir = output_dir + 'train_results/'
    trained_sess_dir = output_dir + 'trained_sess/'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(trained_sess_dir):
        os.makedirs(trained_sess_dir)

    # Output file paths
    train_log_file = train_dir + 'train_log_Dlr={}_Glr={}_beta1={}_reg={}_loss={}.txt'.format(D_lr, G_lr, beta1, reg, loss)
    train_img_file = train_dir + 'train_gen_examples_epoch_'
    val_log_file = val_dir + 'val_log_Dlr={}_Glr={}_beta1={}_reg={}_loss={}.txt'.format(D_lr, G_lr, beta1, reg, loss)
    val_csv_file = val_dir + 'val_metrics_Dlr={}_Glr={}_beta1={}_reg={}_loss={}'.format(D_lr, G_lr, beta1, reg, loss)

    # Initialize the log files
    start_msg = local_clock() + '  Started training model with D_lr={}, G_lr={}, beta1={}, reg={}\n'.format(D_lr, G_lr, beta1, reg)
    print(start_msg)
    with open(train_log_file, 'w') as handle:
        handle.write(start_msg)
        handle.write('device={}\n'.format(device))
    with open(val_log_file, 'w') as handle:
        handle.write(start_msg)
        handle.write('device={}\n'.format(device))

    # Get the data set
    train_gray_dir = train_data_dir + 'gray/'
    train_color_dir = train_data_dir + 'color/'
    val_gray_dir = val_data_dir + 'gray/'
    val_color_dir = val_data_dir + 'color/'
    train_data = Dataset(train_gray_dir, train_color_dir, batch_size, img_dim, shuffle=True)
    train_example_data = Dataset(train_gray_dir, train_color_dir, batch_size, img_dim, shuffle=False)
    val_data = Dataset(val_gray_dir, val_color_dir, batch_size, img_dim, shuffle=False)
    
    # Construct computational graph
    tf.reset_default_graph() # reset the graph
    with tf.device(device):
        is_training = tf.placeholder(tf.bool, name='is_training')
        gray_img = tf.placeholder(tf.float32, [None, img_dim, img_dim, 1])
        color_img = tf.placeholder(tf.float32, [None, img_dim, img_dim, 4])

        pair_real = tf.concat([gray_img, color_img], axis=3)
        G_sample = generator(gray_img, is_training)
        pair_fake = tf.concat([gray_img, G_sample], axis=3)

        with tf.variable_scope('') as scope:
            logits_real = discriminator(pair_real, is_training)
            scope.reuse_variables()
            logits_fake = discriminator(pair_fake, is_training)

        # Get the list of trainable variables for the discriminator and generator
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        # Get solvers
        D_solver, G_solver = get_solvers(D_lr=D_lr, G_lr=G_lr, beta1=beta1)

        # Compute the losses
        D_loss, G_loss = gan_loss(logits_real, logits_fake)
        img_loss = loss_method(G_sample, color_img, reg=reg)

        # Calculate the MSE between generated images and original color images
        mse = calculate_mse(G_sample, color_img)

        # Set up the training operations
        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
        with tf.control_dependencies(D_update_ops):
            D_train_op = D_solver.minimize(0*D_loss, var_list=D_vars)

        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
        with tf.control_dependencies(G_update_ops):
            G_train_op = G_solver.minimize(0*G_loss + img_loss, var_list=G_vars)

        # Remember the nodes we want to run in the future
        tf.add_to_collection('is_training', is_training)
        tf.add_to_collection('gray_img', gray_img)
        tf.add_to_collection('color_img', color_img)
        tf.add_to_collection('G_sample', G_sample)
        tf.add_to_collection('D_loss', D_loss)
        tf.add_to_collection('G_loss', G_loss)
        tf.add_to_collection('img_loss', img_loss)
        tf.add_to_collection('mse', mse)
        tf.add_to_collection('D_train_op', D_train_op)
        tf.add_to_collection('G_train_op', G_train_op)

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            print(local_clock() + '  Started epoch %d' % (epoch))
            for t, (gray_img_np, color_img_np) in enumerate(train_data):
                gray_processed_np = preprocess(gray_img_np)
                color_processed_np = preprocess(color_img_np)
                feed_dict = {gray_img: gray_processed_np, color_img: color_processed_np, is_training: True}
                _, D_loss_np = sess.run([D_train_op, D_loss], feed_dict=feed_dict)
                _, G_loss_np, img_loss_np = sess.run([G_train_op, G_loss, img_loss], feed_dict=feed_dict)
                mse_np = sess.run(mse, feed_dict=feed_dict)

            # Save the results to the train log file
            epoch_train_time = local_clock() + '\n'
            epoch_train_msg = 'Epoch %d  D loss: %0.4f  G loss: %0.4f  img loss: %0.4f  MSE: %0.4f' % (epoch, D_loss_np, G_loss_np, img_loss_np, mse_np)
            print(local_clock() + '  ' + epoch_train_msg)
            epoch_train_msg += '\n'
            with open(train_log_file, 'a') as handle:
                handle.write('\n')
                handle.write(epoch_train_time)
                handle.write(epoch_train_msg)

            # Save examples of generated images
            for j, (gray_example_np, color_example_np) in enumerate(train_example_data):
                gray_example_processed_np = preprocess(gray_example_np)
                color_example_processed_np = preprocess(color_example_np)
                break # only load the first batch as examples
            example_feed_dict = {gray_img: gray_example_processed_np, 
                                 color_img: color_example_processed_np, 
                                 is_training: False}
            gen_example_np = sess.run(G_sample, feed_dict=example_feed_dict)
            gen_example_np = postprocess(gen_example_np)
            show_images(gen_example_np, post_process=False, save=True, filepath=train_img_file + str(epoch) + '.png')

            # If true, evaluate on the validation data set
            if eval_val:
                val_log_note = 'Epoch ' + str(epoch)
                epoch_val_img_dir = val_img_dir + 'epoch' + str(epoch) + '/'
                if not os.path.exists(epoch_val_img_dir):
                    os.makedirs(epoch_val_img_dir)
                epoch_val_csv = val_csv_file + '_epoch' + str(epoch) + '.csv'
                evaluate_model(sess=sess,
                               graph_gray=gray_img, 
                               graph_color=color_img, 
                               graph_training=is_training,
                               graph_D_loss=D_loss, 
                               graph_G_loss=G_loss, 
                               graph_img_loss=img_loss, 
                               graph_G_sample=G_sample, 
                               dataset=val_data, 
                               log_filename=val_log_file, 
                               log_note=val_log_note, 
                               csv_filename=epoch_val_csv, 
                               output_imgs=save_eval_img, 
                               img_dir=epoch_val_img_dir, 
                               num_eval_img=num_eval_img)

            # Save the session when the epoch is done
            saver = tf.train.Saver()
            sess_name = 'Dlr={}_Glr={}_beta1={}_reg={}_loss={}_epoch_{}'.format(D_lr, G_lr, beta1, reg, loss, epoch)
            sess_file = trained_sess_dir + sess_name
            saver.save(sess, sess_file)

            print(local_clock() + '  Finished epoch %d' % (epoch))
            print('')
    return

def evaluate_trained_gan(meta_file, checkpoint_path, eval_data_dir, output_dir, num_eval_img=100, batch_size=16, img_dim=256):
    # Set up output directories    
    eval_dir = output_dir + 'eval_results/'
    eval_img_dir = eval_dir + 'imgs/'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(eval_img_dir):
        os.makedirs(eval_img_dir)

    # Output file paths
    eval_log_file = eval_dir + 'eval_log.txt'
    eval_csv_file = eval_dir + 'eval_metrics.csv'

    # Initialize the log file
    start_msg = local_clock() + '  Started evaluating model.'
    with open(eval_log_file, 'w') as handle:
        handle.write(start_msg)
        handle.write('meta file: ' + meta_file + '\n')
        handle.write('checkpoint path: ' + checkpoint_path + '\n')
        handle.write('eval data directory: ' + eval_data_dir + '\n')

    # Get the data set
    eval_gray_dir = eval_data_dir + 'gray/'
    eval_color_dir = eval_data_dir + 'color/'
    eval_data = Dataset(eval_gray_dir, eval_color_dir, batch_size, img_dim, shuffle=False)

    # Restore the trained session and evaluate on the evlation dataset
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_file)
        new_saver.restore(sess, checkpoint_path)

        # Restore the variables
        is_training = tf.get_collection('is_training')[0]
        gray_img = tf.get_collection('gray_img')[0]
        color_img = tf.get_collection('color_img')[0]
        G_sample = tf.get_collection('G_sample')[0]
        D_loss = tf.get_collection('D_loss')[0]
        G_loss = tf.get_collection('G_loss')[0]
        img_loss = tf.get_collection('img_loss')[0]
        mse = tf.get_collection('mse')[0]
        D_train_op = tf.get_collection('D_train_op')[0]
        G_train_op = tf.get_collection('G_train_op')[0]

        evaluate_model(sess=sess, graph_gray=gray_img, graph_color=color_img, graph_training=is_training, 
                       graph_D_loss=D_loss, graph_G_loss=G_loss, graph_img_loss=img_loss, 
                       graph_G_sample=G_sample, dataset=eval_data, 
                       log_filename=eval_log_file, log_note='Finished evaluating.', csv_filename=eval_csv_file, 
                       output_imgs=True, img_dir=eval_img_dir, num_eval_img=num_eval_img)
    return

def resume_train_gan(meta_file, checkpoint_path, train_data_dir, val_data_dir, output_dir, num_epochs,  
                     batch_size=16, eval_val=True, save_eval_img=True, num_eval_img=100, device='/gpu:0', img_dim=256):
    # Set up output directories    
    val_dir = output_dir + 'val_results/'
    val_img_dir = val_dir + 'imgs/'
    train_dir = output_dir + 'train_results/'
    trained_sess_dir = output_dir + 'trained_sess/'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(trained_sess_dir):
        os.makedirs(trained_sess_dir)

    # Get the trained model configuration
    configs = checkpoint_path.split('/')[-1]
    pre_epoch = int(configs.split('_')[-1])
    params_str = configs.split('_')[:-2]
    params_str = '_'.join(params_str)

    # Output file paths
    train_log_file = train_dir + 'train_log_{}.txt'.format(params_str)
    train_img_file = train_dir + 'train_gen_examples_epoch_'
    val_log_file = val_dir + 'val_log_{}.txt'.format(params_str)
    val_csv_file = val_dir + 'val_metrics_{}'.format(params_str)

    # Initialize the log files
    start_msg = local_clock() + '  Resumed training model with {} and {} epochs\n'.format(params_str, pre_epoch)
    print(start_msg)
    with open(train_log_file, 'w') as handle:
        handle.write(start_msg)
        handle.write('device={}\n'.format(device))
    with open(val_log_file, 'w') as handle:
        handle.write(start_msg)
        handle.write('device={}\n'.format(device))

    # Get the data set
    train_gray_dir = train_data_dir + 'gray/'
    train_color_dir = train_data_dir + 'color/'
    val_gray_dir = val_data_dir + 'gray/'
    val_color_dir = val_data_dir + 'color/'
    train_data = Dataset(train_gray_dir, train_color_dir, batch_size, img_dim, shuffle=True)
    train_example_data = Dataset(train_gray_dir, train_color_dir, batch_size, img_dim, shuffle=False)
    val_data = Dataset(val_gray_dir, val_color_dir, batch_size, img_dim, shuffle=False)

    # Restore the trained session and evaluate on the evlation dataset
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_file)
        new_saver.restore(sess, checkpoint_path)

        # Restore the variables
        is_training = tf.get_collection('is_training')[0]
        gray_img = tf.get_collection('gray_img')[0]
        color_img = tf.get_collection('color_img')[0]
        G_sample = tf.get_collection('G_sample')[0]
        D_loss = tf.get_collection('D_loss')[0]
        G_loss = tf.get_collection('G_loss')[0]
        img_loss = tf.get_collection('img_loss')[0]
        mse = tf.get_collection('mse')[0]
        D_train_op = tf.get_collection('D_train_op')[0]
        G_train_op = tf.get_collection('G_train_op')[0]

        for epoch in range(pre_epoch + 1, pre_epoch + 1 + num_epochs):
            print(local_clock() + '  Started epoch %d' % (epoch))
            for t, (gray_img_np, color_img_np) in enumerate(train_data):
                gray_processed_np = preprocess(gray_img_np)
                color_processed_np = preprocess(color_img_np)
                feed_dict = {gray_img: gray_processed_np, color_img: color_processed_np, is_training: True}
                _, D_loss_np = sess.run([D_train_op, D_loss], feed_dict=feed_dict)
                _, G_loss_np, img_loss_np = sess.run([G_train_op, G_loss, img_loss], feed_dict=feed_dict)
                mse_np = sess.run(mse, feed_dict=feed_dict)

            # Save the results to the train log file
            epoch_train_time = local_clock() + '\n'
            epoch_train_msg = 'Epoch %d  D loss: %0.4f  G loss: %0.4f  img loss: %0.4f  MSE: %0.4f' % (epoch, D_loss_np, G_loss_np, img_loss_np, mse_np)
            print(local_clock() + '  ' + epoch_train_msg)
            epoch_train_msg += '\n'
            with open(train_log_file, 'a') as handle:
                handle.write('\n')
                handle.write(epoch_train_time)
                handle.write(epoch_train_msg)

            # Save examples of generated images
            for j, (gray_example_np, color_example_np) in enumerate(train_example_data):
                gray_example_processed_np = preprocess(gray_example_np)
                color_example_processed_np = preprocess(color_example_np)
                break # only load the first batch as examples
            example_feed_dict = {gray_img: gray_example_processed_np, 
                                 color_img: color_example_processed_np, 
                                 is_training: False}
            gen_example_np = sess.run(G_sample, feed_dict=example_feed_dict)
            gen_example_np = postprocess(gen_example_np)
            show_images(gen_example_np, post_process=False, save=True, filepath=train_img_file + str(epoch) + '.png')

            # If true, evaluate on the validation data set
            if eval_val:
                val_log_note = 'Epoch ' + str(epoch)
                epoch_val_img_dir = val_img_dir + 'epoch' + str(epoch) + '/'
                if not os.path.exists(epoch_val_img_dir):
                    os.makedirs(epoch_val_img_dir)
                epoch_val_csv = val_csv_file + '_epoch' + str(epoch) + '.csv'
                evaluate_model(sess=sess,
                               graph_gray=gray_img, 
                               graph_color=color_img, 
                               graph_training=is_training,
                               graph_D_loss=D_loss, 
                               graph_G_loss=G_loss, 
                               graph_img_loss=img_loss, 
                               graph_G_sample=G_sample, 
                               dataset=val_data, 
                               log_filename=val_log_file, 
                               log_note=val_log_note, 
                               csv_filename=epoch_val_csv, 
                               output_imgs=save_eval_img, 
                               img_dir=epoch_val_img_dir, 
                               num_eval_img=num_eval_img)

            # Save the session when the epoch is done
            saver = tf.train.Saver()
            sess_name = params_str + '_epoch_' + str(epoch)
            sess_file = trained_sess_dir + sess_name
            saver.save(sess, sess_file)

            print(local_clock() + '  Finished epoch %d' % (epoch))
            print('')
    return
