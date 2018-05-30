import time
import math
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec

def local_clock(shift=-25200):
	return time.asctime(time.localtime(time.time() + shift))

def chunk(l, chunks):
	chunk_list = []
	chunk_size = math.floor(len(l) / chunks)
	for i in range(chunks - 1):
		chunk_list.append(l[chunk_size*i:chunk_size*(i+1)])
	chunk_list.append(l[chunk_size*(chunks - 1):])
	return chunk_list

def preprocess(img):
    img = (img / 255.0) * 2 - 1
    return img

def postprocess(img):
    img = (img + 1) / 2 * 255.0
    return img

def show_images(images, post_process=False, save=False, filepath=None):
    N, H, W, C = images.shape
    sqrtN = int(np.ceil(np.sqrt(N)))
    if post_process:
        images = postprocess(images)
    images = images.astype(np.uint8)
    fig = plt.figure(figsize=(sqrtN, sqrtN))
    gs = gridspec.GridSpec(sqrtN, sqrtN)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i in range(N):
        img = images[i,:,:,:]
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if C == 1:
            plt.imshow(img.reshape((H, W)), cmap='gray')
        else:
            plt.imshow(img)
    if save and filepath != None:
        fig.savefig(filepath)
    plt.close(fig)
    return

def save_image(image, filepath, post_process=False, width=4, height=4):
    H, W, C = image.shape
    if post_process:
        image = postprocess(image)
    image = image.astype(np.uint8)
    fig = plt.figure(figsize=(width, height))
    plt.axis('off')
    if C == 1:
        plt.imshow(image.reshape((H, W)), camp='gray')
    else:
        plt.imshow(image)
    fig.savefig(filepath)
    plt.close(fig)
    return