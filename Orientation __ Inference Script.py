#!/usr/bin/env python
# coding: utf-8

# -> In this code, to save memory, I have read and stored the directories of all the images at 256x256 in a numpy array.
# 
# -> 21000 images : 30 images are being taken from every folder. The first one is the face-on. So 29 unique oriented images, at different inclination angles and posang. 30x700 = 21000 images (SOURCE IMAGES). Corresponding to every source image, I have the corresponding face-on image, so that's another array with directories of 21000 images (TARGET IMAGES).
# 
# -> The source images are then being deployed and generated images are obtained, and compared with the corresponding target image. The network is being trained to generate images similar to their corresponding target image. 
# 
# -> The summarize_performance function is called at certain intervals to save a version of the model and a plot at that point in time. I am saving them inside a folder which happens to be unique for each entire run. The creation and naming of folders is done automatically.
# 
# -> Finally, we may choose any of the saved models' name and deploy it to give us the desired result.
# 
# ____

# # Orienting Network to find generate face-on images for PPDs 
# ## SOURCE CODE

#     This notebook is used to train a CGAN (Conditional Generative Adversarial Network), incorporating 
#     the Pix2Pix concept on a dataset of Proto-planetary Disk images obtained from the FARGO3D simulations.
# 
#     
#     Summary       : We are trying to generate face-on images of Protoplanetary Disks from images of any 
#                     orientation.
# 
# 
#     Code&Config   : The code is being done on Jupyter Notebook platform, and is being run on MacOS 13.1, 
#                     Apple M1, 8gb configuration.  
#                
#          
# ________________

# ### Supervisor   : Dr. Sayantan Auddy
# #### Written by    : Dyutiman Santra
# #### Created       : 18th February, 2024
# 
# _______________________________________

# ||  Importing Modules  ||

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob 

#*****************

from tensorflow.keras import layers, losses

from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape, Dropout, LeakyReLU, Embedding, Concatenate
from keras.models import Model
from keras.losses import binary_crossentropy

from keras.initializers import RandomNormal
from keras.layers import BatchNormalization
from keras.layers import Activation

from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model

from sklearn.model_selection import train_test_split

#*****************

from numpy import asarray
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from numpy.random import randn
from numpy import vstack

#*****************

import time
import sys

import random
from datetime import datetime as dt
from IPython import display
from PIL import Image

#*****************

# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error as mse
# # from skimage.metrics import normalized_cross_correlation as ncc
# from skimage.metrics import normalized_root_mse as nrmse
# from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import convolve


# ||  Checking the availbale number of GPUs  ||

# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:",gpu.name,"Type:",gpu.device_type)
print("TensorFlow version used ",tf.__version__)


# _______

# 
# # Specifications

# In[3]:


# To specify our requirements

no_o_folders = 7
pxl = 64 # to set the desired pixel value


# ||  Reading DATA csv ||

# In[4]:


df = pd.read_csv('/Users/Dyutiman/Documents/ML_Project/Pix2Pix/RT_Dataset_incl_posang.csv')  #path of csv file

# print("The dataframe is:") #displaying csv
print(df,"\n")


Ind, X_inlabel, Y_inlabel = [], [], []

Ind = df["index"] #to store the image numbers excluding translational changes
X_label = df["incl"] #to store the inclination angle
Y_label = df["posang"] #to store the position angle

print("The dataframe is loaded.") #displaying csv


# ___________

# ||  Reading and loading DATA images ||

# In[5]:


## Reading the Image Dataset, from specified folders

path = "/Users/Dyutiman/Documents/ML_Project/Sayantan Da Projects/Final 1.5l" #specifying the path of the dataset

X = [] 															# a List to store oriented images
Y = [] 															# a List to store face-on image

k=0
m=0

print(f"Total number of folders to be loaded is {no_o_folders}.\n")

for i in range(1, no_o_folders+1):

    directory =path+"/RT_A_"+ str(i)+"/*.png"
    data_set_indiv = glob.glob(directory)

    loc = path+"/RT_A_"+ str(i)
    
    for j in Ind: #Loading the oriented images in X and correspondsing face-on images in Y
        X.append(loc+"/image_"+str(j)+".png")
        Y.append(loc+"/image_1.png")
        m = m+1
    
    k = k+1
    display.clear_output(wait=True)
    print("Total count of images = %d. ----> %2.2f %s"%(m,(k/no_o_folders)*100,'%'))
    
print("*End of code block*")

print(len(X), "image diretories are loaded.")
# print(Y)


# ____

# In[6]:


# To display all the loaded images 
if (False):
    fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))

    k=0

    for i in trainX:
#         print(i)
        try:
            axes.set_title("Run:{}".format(k))
            plt.imshow(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB)) #, cmap='RdGy')
            #print(k)
            k=k+1
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.001)
            if(k%5 == 0):
                fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))
        except KeyboardInterrupt:
            break


# _________________________________

# # Load and start !!!

# In[7]:


# deallocating the unreferenced objects and freeing up memory (OPTIONAL)

import gc
gc.collect()


# In[8]:


# Splitting the dataset

trainX, testX, trainy, testy = train_test_split(X, Y,
                                   random_state=42, 
                                   test_size=0.10,
                                   shuffle=True)

# summarize the shape of the dataset
print('Train', len(trainX), '\nTest', len(testX), '\nTrainLabel', len(trainy), '\nTestLabel', len(testy))


# _________________________________

# ### Moving forward with the cGAN Network

# ## Network Setup

# In[9]:


def define_discriminator(image_shape):
    
    # weight initialization
    init = RandomNormal(stddev=0.02) #As described in the original paper
    
    # source image input
    in_src_image = Input(shape=image_shape)  #Image we want to convert to another image
    # target image input
    in_target_image = Input(shape=image_shape)  #Image we want to generate after training. 
    
    # concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    
    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
#     print("Inside discriminator, 1st 512",d)
    # C512 2nd: 4x4 kernel Stride 2x2 
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
#     print("Inside discriminator, 2nd 512",d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
#     print("Inside discriminator, 3rd 512",d)
    
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    
    #The model is trained with a batch size of one image and Adam opt. with a small learning rate and 0.5 beta. 
    #The loss for the discriminator is weighted by 50% for each model update.
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# In[10]:


# Defining the generator - a U-net
# defining an encoder block to be used in generator
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    #print(layer_in.shape)
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model - U-net
def define_generator(image_shape=(pxl,pxl,1)):      #(256,256,3)): #to change depending on input
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)

    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
#     print("Inside define_generator , 1st 256",e3)
    e4 = define_encoder_block(e3, 512)
#     print("Inside define_generator , 1st 512",e4)
    e5 = define_encoder_block(e4, 512)
#     print("Inside define_generator , 2nd 512",e5)
#     e6 = define_encoder_block(e5, 512)
#     print("Inside define_generator , 3rd 512",e6)
#     time.sleep(3)
#     e7 = define_encoder_block(e6, 512) #this should be included if images are 256x256 (e6, e7, b,d1, d2 should be accordingly placed in places)
#     print("Inside define_generator , 4th 512",e7)
#     time.sleep(3)
    #   e8 = define_encoder_block(e7, 512) #this should be included if images are 512x512
    print("Working till encoder block")
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5) #this would be (e7) in case of 256x256
    b = Activation('relu')(b)
#     print("Inside define_generator , final",b)

#   d0 = decoder_block(b, e8, 512) #this should be included if images are 512x512
#     d1 = decoder_block(b, e7, 512) #this should be included if images are 256x256
#     d2 = decoder_block(d1, e6, 512) #the b will change to d1 if resolution is increased
    d3 = decoder_block(b, e5, 512) 
#     print("***Decoded, after 1st 512", d3)
    d4 = decoder_block(d3, e4, 512, dropout=False)
#     print("***Decoded, after 2nd 512", d4)
    d5 = decoder_block(d4, e3, 256, dropout=False)
#     print("***Decoded, after 256", d5)
    d6 = decoder_block(d5, e2, 128, dropout=False)
#     print("***Decoded, after 128", d6)
    d7 = decoder_block(d6, e1, 64, dropout=False)
#     print("***Decoded, after 64", d7)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7) #Modified 
#     print("***Decoded, final g", g)
    out_image = Activation('tanh')(g)  #Generates images in the range -1 to 1. So we change inputs also to -1 to 1
    # define model
    model = Model(in_image, out_image)
    return model


# In[11]:


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False      
            
    # define the source image
    in_src = Input(shape=image_shape)
    # suppy the image as input to the generator 
    gen_out = g_model(in_src)
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    
    #Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    #Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model


# In[12]:


# selecting a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix//29]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# In[13]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# In[ ]:


# defining input shape based on the loaded dataset

# image_shape = trainX.shape[1:]
image_shape = (pxl,pxl,1)
print("Shape is",image_shape)

# define the models
print("Working so far : d")
d_model = define_discriminator(image_shape)

print("Working so far : g")
g_model = define_generator(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# Defining dataset, loading and prepariing training images
dataset = [trainX, trainy]


# ### Network Setup complete.

# _________________________________

# # Loading weights

# In[20]:


model_path = '/Users/Dyutiman/Documents/ML_Project/Pix2Pix/ModelWeight24-05-04_12_12/modelWeight_0000260_24-05-04_12_13.h5'

g_model.load_weights(model_path)


# _________________________________

# # RESULT !

# In[21]:


from scipy.ndimage import gaussian_filter

def COMPUTE_DIFF(xx, gen_img):
#     gray_image1 = xx #cv2.cvtColor(xx) #, cv2.COLOR_BGR2GRAY)
#     gray_image2 = gen_img #cv2.cvtColor(gen_img) #, cv2.COLOR_BGR2GRAY)
    
    gray_image1 = (xx + 1) / 2
    gray_image2 = (gen_img + 1) / 2

#     print("gray_image1 shape")
#     print(gray_image1)
#     print(gray_image1.shape)
#     print(gray_image2.shape)
    
    # Calculate Mean Squared Error (MSE)
    mse_value = np.mean((gray_image1 - gray_image2) ** 2)

    # Calculate Structural Similarity Index (SSIM)without using any library:
#     def ssim(image1, image2):
#         K1 = 0.01
#         K2 = 0.03
    
#         # Compute the means
#         mean_x = np.mean(image1)
#         mean_y = np.mean(image2)

#         # Compute the variances
#         var_x = np.var(image1)
#         var_y = np.var(image2)

#         # Compute the covariance
#         cov_xy = np.mean((image1 - mean_x) * (image2 - mean_y))

#         # Compute the SSIM index
#         ssim_index = (2 * mean_x * mean_y + K1) * (2 * cov_xy + K2) / ((mean_x ** 2 + mean_y ** 2 + K1) * (var_x + var_y + K2))

#         return ssim_index


    def ssim(image1, image2, window_size=1, sigma=1.5):
        K1 = 0.01
        K2 = 0.03

        # Apply Gaussian weighting window
        gaussian_window = gaussian_filter(np.ones((window_size, window_size)), sigma)

        # Compute the means of the images using the Gaussian-weighted average
        mean_x = gaussian_filter(image1, sigma) / gaussian_window.sum()
        mean_y = gaussian_filter(image2, sigma) / gaussian_window.sum()

        # Compute the variances of the images
        var_x = gaussian_filter(image1 ** 2, sigma) / gaussian_window.sum() - mean_x ** 2
        var_y = gaussian_filter(image2 ** 2, sigma) / gaussian_window.sum() - mean_y ** 2

        # Compute the covariance between the images
        cov_xy = (gaussian_filter(image1 * image2, sigma) / gaussian_window.sum()) - (mean_x * mean_y)

        # Compute the SSIM index
        num = (2 * mean_x * mean_y + K1) * (2 * cov_xy + K2)
        den = (mean_x ** 2 + mean_y ** 2 + K1) * (var_x + var_y + K2)
        ssim_index = np.mean(num / den)  # Compute mean SSIM index over the entire image

        return ssim_index



    ssim_value = ssim(gray_image1, gray_image2)

#     #Print the similarity metrics
#     print(f"SSIM: {ssim_value:.4f}")
#     print(f"MSE: {mse_value:.4f}")
    return ssim_value, mse_value


# In[22]:


# def COMPUTE_DIFF(xx, gen_img):
#     gray_image1 = xx
#     gray_image2 = gen_img

#     # Calculate Mean Squared Error (MSE)
#     mse_value = np.mean((gray_image1 - gray_image2) ** 2)

#     # Calculate Structural Similarity Index (SSIM)
#     ssim_value = ssim(gray_image1, gray_image2)

#     return ssim_value, mse_value

# def _gaussian_window(window_size, sigma):
#     """
#     Generate a 2D Gaussian window for convolution.
#     """
#     x = np.arange(0, window_size, 1, float)
#     y = x[:, np.newaxis]
#     x0 = y0 = window_size // 2
#     gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#     return gaussian / np.sum(gaussian)

# def ssim(image1, image2, window_size=11, sigma=1.5):
#     K1 = 0.01
#     K2 = 0.03
    
#     # Generate Gaussian window
#     window = _gaussian_window(window_size, sigma)
    
#     # Compute means
#     mu1 = convolve(image1, window, mode='constant')
#     mu2 = convolve(image2, window, mode='constant')

#     # Compute variances and covariance
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = convolve(image1 ** 2, window, mode='constant') - mu1_sq
#     sigma2_sq = convolve(image2 ** 2, window, mode='constant') - mu2_sq
#     sigma12 = convolve(image1 * image2, window, mode='constant') - mu1_mu2

#     # Compute SSIM
#     num = (2 * mu1_mu2 + K1) * (2 * sigma12 + K2)
#     den = (mu1_sq + mu2_sq + K1) * (sigma1_sq + sigma2_sq + K2)

#     ssim_map = num / den
#     ssim_index = np.mean(ssim_map)

#     return ssim_index


# In[23]:


def dataset_batch_test(lower, upper):
    TX = []
    TY = []

    for i in range(lower, upper):
        image_dir = testX[i]
        emag = cv2.imread(image_dir, 0)
        emag = np.expand_dims(emag, axis=-1)
        TX.append(emag[57:428, 106:477])
        
        trg_dir = testy[i]
        emag = cv2.imread(trg_dir, 0)
        emag = np.expand_dims(emag, axis=-1)
        TY.append(emag[57:428, 106:477])
        
        if((i+1)%50 == 0):
#             display.clear_output(wait=True)
            print("Loading batch %.2f%s" % ((i+1-lower)/(upper-lower)*100,"%"))
    
    ttxx = np.asarray(TX)
    ttyy = np.asarray(TY)

    
    tx = tf.image.resize(ttxx, [pxl, pxl])
    ty = tf.image.resize(ttyy, [pxl, pxl])

    T_X = tx.numpy()
    T_Y = ty.numpy()
    
    # scale from [0,255] to [-1,1]
    T_X = (T_X - 127.5) / 127.5
    T_Y = (T_Y - 127.5) / 127.5

    return [T_X,T_Y]


# ________________________________________________

# # Generating face-on images

# In[25]:


# plotting source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
#     print(np.min(src_img))
#     print(np.min(gen_img))
#     print(np.min(tar_img))
    titles = ['Source', 'Generated', 'Expected']
    
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i]) #, cmap='RdGy')
        # show title
        plt.title(titles[i])
        if( i==1 ):
            score, diff = COMPUTE_DIFF(images[1], images[2])
            print(score, diff)
            trunc_no = str(score*10**2)
            plt.title(f"SSIM = {score*10**2} \nMSE = {diff*10**2:.2f} \n Generated")
#             plt.title(f"SSIM = {trunc_no[6:12]} (7-dec) \nMSE = {diff*10**2:.2f} \n Generated")

        

[X1, X2] = dataset_batch_test(0,len(testX))

print(X1.shape, X2.shape)

for i in range(10):
    fig, axes = plt.subplots(ncols=3, sharex=False,sharey=True, figsize=(12, 8))
    
    # selecting random example
    ix = randint(0, len(X1), 1)
    print(ix)
    # generate image from source
    gen_image = g_model.predict(X1[ix])
#     print(gen_image)
    # plot all three images
    
#     score, diff = COMPUTE_DIFF(X1[ix], gen_image)
#     plt.title(score)
    
    plot_images(X1[ix], gen_image, X2[ix])

plt.show()

## We may use the previously generated CSV to choose images of specific orientation as our Source Images
    


# _________________________

# # Plots for statistical analysis

# In[28]:


def getplot():
    
    [X1, X2] = dataset_batch_test(0,len(testX))

    print(X1.shape, X2.shape)
    SSIM = []
    MSE = []

    for i in range(len(X2)):
        gen_image = g_model.predict(X1[[i]])

        ind1, ind2 = COMPUTE_DIFF(gen_image, X2[i])

        display.clear_output(wait=True)
        print(f"{(i+1)*100//len(X2)}%")
        print(ind1)
        
        SSIM.append(ind1)
        MSE.append(ind2)

    ## We may use the previously generated CSV to choose images of specific orientation as our Source Images
    return SSIM, MSE


    # 1. **Histogram:**
    #    - A histogram will give us a visual representation of the frequency distribution of the SSIM index values. This will help you understand how the values are spread across different ranges.

    # 2. **Box plot:**
    #    - A box plot (also known as a box-and-whisker plot) will show us the distribution of the SSIM index values, including the median, quartiles, and any outliers. This is useful for understanding the central tendency and variability of the data.

    # 3. **Violin plot:**
    #    - A violin plot combines aspects of a box plot and a kernel density plot to provide a more comprehensive view of the distribution of the SSIM index values. It can show both the summary statistics and the underlying probability density of the data.

    # 4. **Scatter plot:**
    #    - If we have paired data or want to explore relationships between different variables, a scatter plot can be useful. We can plot the SSIM index values against another variable (if available) to see if there are any patterns or correlations.


# In[29]:


SSIM, MSE = getplot()
len(SSIM)


# In[31]:


# Plotting a histogram
plt.hist(SSIM, bins=2100, edgecolor='white', alpha=0.7, label='SSIM', color= 'teal') #'yellowgreen')
plt.xlim(.99, 1.001)
plt.xlabel('SSIM Index Values')
plt.ylabel('Frequency')
plt.title('Histogram of SSIM Index Values')
plt.show()

# Plotting a box plot
plt.boxplot(SSIM)
plt.ylim(.98, 1.00)
plt.ylabel('SSIM Index Values')
plt.title('Box Plot of SSIM Index Values')
plt.show()

# # Plotting a violin plot
# plt.violinplot(SSIM, showmeans=True)
# plt.ylabel('SSIM Index Values')
# plt.title('Violin Plot of SSIM Index Values')
# plt.show()

# # paired data or another variable to plot against SSIM index values, use a scatter plot

# plt.scatter(SSIM, testX)
# plt.xlabel('SSIM Index Values')
# plt.ylabel('Paired Data')
# plt.title('Scatter Plot of SSIM Index Values vs. Paired Data')
# plt.show()


# In[32]:


# Plotting a histogram
plt.hist(MSE, bins=3150, edgecolor='white', alpha=0.7, label='MSE', color= 'teal') #'yellowgreen')
plt.xlim(.0, .0015)
plt.xlabel('MSE Index Values')
plt.ylabel('Frequency')
plt.title('Histogram of MSE Index Values')
plt.show()

# Plotting a box plot
plt.boxplot(MSE)
plt.ylim(.0, .003)
plt.ylabel('MSE Index Values')
plt.title('Box Plot of MSE Index Values')
plt.show()


# ________________________

# # Plotting Real Images

# In[34]:


def hello_universe(location_of_image):
    
#     location_of_image = "/Users/Dyutiman/Downloads/Original Disk Images/"+true_dir #HD-163296.webp"

    real_image = [np.expand_dims(cv2.imread(location_of_image, 0), axis=-1)]
    RI = tf.image.resize(np.asarray(real_image), [pxl, pxl])
    real_image = RI.numpy()
    
    # scale from [0,255] to [-1,1]
    real_image = (real_image - 127.5) / 127.5


    def plot_images_real(src_img, gen_img):
        images = vstack((src_img, gen_img))
        # scale from [-1,1] to [0,1]
        images = (images + 1) / 2.0
        titles = ['Source', 'Generated', 'Expected']
        
        to_return_im = []
        to_return_tl = []
    
        # plot images row by row
        for i in range(len(images)):
            # plot raw pixel data
            to_return_im.append(images[i]*255)
            # show title
            to_return_tl.append(titles[i])
            
        return to_return_im, to_return_tl

    # generate image from source
    gen_image = g_model.predict(real_image)
    return plot_images_real(real_image, gen_image)
    
    
    
list_direc = glob.glob("/Users/Dyutiman/Downloads/Original Disk Images/*")
# print(list_direc)
col_cntrl = 0
c = []
d = []

for i in list_direc:
    
    a,b = hello_universe(i)
    c.extend(a)
    d.extend(b)
    col_cntrl = col_cntrl+1
    
    if(col_cntrl%2 == 0):
        fig, axes = plt.subplots(ncols=4, sharex=False,sharey=True, figsize=(12, 10))
        for i in range(4):
            plt.subplot(1, 4, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(c[i])
            # show title
            plt.title(d[i])
        c = []
        d = []
    
plt.show()


# ________________________

# ###### Preparing CSV for the code 

# In[ ]:


import csv

incl_list = [0., 15., 30., 45., 60.]
# phi_list = [0.]
posang_list = [0., 15., 30., 45., 60., 75.]
pointaux_list = [0., 2.5, 5.0, 7.5, 10.0]

csvvv = []

f=1
for i in incl_list:
#     for j in phi_list:
        for k in posang_list:
            for l in pointaux_list:
                if(l==0):
                    csvvv.append([f, i, k])
                f=f+1

with open('/Users/Dyutiman/Documents/ML_Project/Pix2Pix/RT_Dataset_incl_posang.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index', 'incl', 'posang'])  # Write header
    writer.writerows(csvvv)  # Write data rows


# ________________________

# End of code!
