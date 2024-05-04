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

# In[6]:


# To specify our requirements

no_o_folders = 1
pxl = 256 # to set the desired pixel value


# ||  Reading DATA csv ||

# In[7]:


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

# In[8]:


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

# In[9]:


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

# In[10]:


# deallocating the unreferenced objects and freeing up memory (OPTIONAL)

import gc
gc.collect()


# In[11]:


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

# In[12]:


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
    # C512 2nd: 4x4 kernel Stride 2x2 
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
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


# In[13]:


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


# In[14]:


from keras.layers import Multiply, Add, Reshape, Conv2D, RepeatVector, Concatenate, Activation, Input, Conv2DTranspose, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
import keras.backend as K

# Attention gate
def attention_gate(x, g, inter_channels):
    # Inter-channel attention
    theta_x = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, kernel_size=1, strides=1, padding='same')(f)
    rate = Activation('sigmoid')(psi_f)

    # Apply attention gate
    attention = Multiply()([x, rate])

    return attention

# Define encoder block with attention
def define_encoder_block_with_attention(layer_in, n_filters, batchnorm=True):
    # Encoder block as before
    g = define_encoder_block(layer_in, n_filters, batchnorm=batchnorm)
    # Downsample input tensor to match the spatial dimensions of attention gate output
    skip_in = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same')(layer_in)
    # Apply attention mechanism
    attention = attention_gate(skip_in, g, n_filters // 2)
    # Concatenate attention and encoder block output
    g = Concatenate()([g, attention])
    return g

# Define decoder block with attention
def decoder_block_with_attention(layer_in, skip_in, n_filters, dropout=True):
    # Decoder block as before
    g = decoder_block(layer_in, skip_in, n_filters, dropout=dropout)
    # Apply attention mechanism
    attention = attention_gate(g, skip_in, n_filters // 2)
    # Concatenate attention and decoder block output
    g = Concatenate()([g, attention])
    return g

# Define the standalone generator model - U-net with attention
def define_generator_with_attention(image_shape=(pxl, pxl, 1)):
    # Same as before
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    e1 = define_encoder_block_with_attention(in_image, 64, batchnorm=False)
    e2 = define_encoder_block_with_attention(e1, 128)
    e3 = define_encoder_block_with_attention(e2, 256)
    e4 = define_encoder_block_with_attention(e3, 512)
    e5 = define_encoder_block_with_attention(e4, 512)
#     e6 = define_encoder_block_with_attention(e5, 512)
#     e7 = define_encoder_block_with_attention(e6, 512)

    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)

#     d1 = decoder_block_with_attention(b, e7, 512)
#     d2 = decoder_block_with_attention(d1, e6, 512)
    d3 = decoder_block_with_attention(b, e5, 512)
    d4 = decoder_block_with_attention(d3, e4, 512, dropout=False)
    d5 = decoder_block_with_attention(d4, e3, 256, dropout=False)
    d6 = decoder_block_with_attention(d5, e2, 128, dropout=False)
    d7 = decoder_block_with_attention(d6, e1, 64, dropout=False)

    g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


# In[15]:


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


# In[16]:


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


# In[17]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# ### Network Setup complete.

# _________________________________

# ## Preparing Training Block

# In[18]:


def dataset_batch(lower, upper):
    TX = []
    TY = []

    for i in range(lower, upper):
        image_dir = trainX[i]
        emag = cv2.imread(image_dir, 0)
        emag = np.expand_dims(emag, axis=-1)
        TX.append(emag[57:428, 106:477])
        
        trg_dir = trainy[i]
        emag = cv2.imread(trg_dir, 0)
        emag = np.expand_dims(emag, axis=-1)
        TY.append(emag[57:428, 106:477])
        
        if((i+1)%10 == 0):
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


# In[19]:


# Generating samples and saving plot and the model
# GAN models do not converge, we just want to find a good balance between the generator and the discriminator. 
# Therefore, it makes sense to periodically save the generator model and check how good the generated image looks. 
def summarize_performance(step, g_model, dataset, direct, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB] = dataset
    
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    
    # scale all pixels from [-1,1] to [0,1]
    X_fakeB = (X_fakeB + 1) / 2.0
    
    # plot real source images
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_realA[i]*255)
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fakeB[i]*255)
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(X_realB[i]*255)
    
    # save plot to file
    formatted_date = (dt.now()).strftime("%Y-%m-%d %H:%M:%S")
    FD = formatted_date[2:10]+"_"+formatted_date[11:13]+"_"+formatted_date[14:16]
        
    filename1 = 'ATTN_plot_%07d_%s.png' % ((step+1), FD)
    plt.savefig(f"{direct}/{filename1}")
    plt.close()
    # save the generator model
    filename2 = 'ATTN_modelWeight_%07d_%s.h5' % ((step+1), FD)
    g_model.save_weights(f"{direct}/{filename2}") #CHANGEDCHANGEDCHANGEDCHANGEDCHANGEDCHANGEDCHANGEDCHANGED
    print('>Saved: %s and %s' % (filename1, filename2))


# In[20]:


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch): #usually 100 epochs
    
    start_time = time.time()
    last_time = start_time
    space = " "
    lowest_rec = 100
    highest_rec = 0
    
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainX) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    direct = ""      # to set up directory
    boolean = True   # flag variable to make sure that the directory is set only once
    
    counting_batch_per_epoch = 0
    dataset_index = 0
    
    # manually enumerate epochs
    for i in range(n_steps):
        
#         print(i)
        print(f"{space*65}______________________________________\n")
        print(f"{space*65}Epoch                    : {i//bat_per_epo +1} / {n_epochs}")
        print(f"{space*65}Steps per epoch          : {counting_batch_per_epoch} / {bat_per_epo}")
        
        counting_batch_per_epoch=counting_batch_per_epoch+1
        if(counting_batch_per_epoch==bat_per_epo):
            dataset_index = 0
            counting_batch_per_epoch = 0
        
        #taking batch
        dataset = dataset_batch(dataset_index, dataset_index+n_batch)
        dataset_index = dataset_index+n_batch
        
        
        # select a batch of real samples
        [X_realA, X_realB], y_real = dataset, ones((n_batch, n_patch, n_patch, 1)) #generate_real_samples(dataset, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        
        # summarize performance
#         if((i+1)%100 == 0):
        display.clear_output(wait=True) # ensures a single line print instead of multiple lines (OPTIONAL)
        print(('>%d, d1[%.3f] d2[%.3f] g[%.3f] --->%2.2f %s ' % (i+1, d_loss1, d_loss2, g_loss, (i+1)/n_steps*100, '%')))
        
        #_________________________________________________________TIME...______
        current_time = time.time()
        current_time_elapsed = current_time - start_time
        time_left = (current_time - last_time)*(n_steps-i-1)
        print(f"{space*65}Time Elaspsed{space*12}: {int(current_time_elapsed//3600)}h {int((current_time_elapsed%3600)//60)}m {int(current_time_elapsed%60)}s\n{space*65}Approximate Time Left{space*4}: {int(time_left//3600)}h {int((time_left%3600)//60)}m {int(time_left%60)}s")
        
        
        if((current_time - last_time)>highest_rec): highest_rec = current_time - last_time
        if((current_time - last_time)<lowest_rec): lowest_rec = current_time - last_time
        
        print(f"{space*65}______________________________________\n")
        print(f"{space*65}\033[1mTIME PER STEP (with Attention Mechanism)\033[0m")
        print(f"%sLowest recorded          : %.2fs"%(space*65,lowest_rec))
        print(f"%sCurrent time          ---> \033[1m%.2fs\033[0m"%(space*65,current_time - last_time))
        print(f"%sHighest recorded         : %.2fs"%(space*65,highest_rec))
        
        last_time = current_time
        #______________________________________________________________________
        
        # summarize model performance
        if (i+1) % (bat_per_epo * 5) == 0: # saving model and plot at every 5th epoch
            print('Summarizing and Saving')
            if(boolean):
                print("Setting up directory...")
                formatted_date = (dt.now()).strftime("%Y-%m-%d %H:%M:%S")
                FD = "ModelWeightATTN"+formatted_date[2:10]+"_"+formatted_date[11:13]+"_"+formatted_date[14:16]
                direct = "/Users/Dyutiman/Documents/ML_Project/Pix2Pix/%s"%(FD)
                os.mkdir(direct) #creating directory
                boolean = False
            summarize_performance(i, g_model, dataset, direct)
            gc.collect() #freeing up memory (OPTIONAL)
            
            
#_____________ALERT_________________# auditory indication of the end of execution (OPTIONAL)
    from IPython.display import Audio
    sound_file = "/Users/Dyutiman/Downloads/terminating alarm.mp3"
    Audio(sound_file, autoplay=True)
    


# In[21]:


# define input shape based on the loaded dataset

# image_shape = trainX.shape[1:]
image_shape = (pxl,pxl,1)
print("Shape is",image_shape)

# define the models
print("Working so far : d")
d_model = define_discriminator(image_shape)

print("Working so far : g")
g_model = define_generator_with_attention(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# Defining dataset, loading and prepariing training images
# dataset1 = [trainX, trainy]


# def preprocess_data(dataset1):
#     # load compressed arrays
#     # unpack arrays
#     X1, X2 = dataset1[0], dataset1[1]
#     # scale from [0,255] to [-1,1]
#     X1 = (X1 - 127.5) / 127.5
#     X2 = (X2 - 127.5) / 127.5
#     return [X1, X2]

# dataset = preprocess_data(dataset1)


# ### Training Block preparation complete.

# __________________________________________________________________

# # TRAINING !

# In[22]:


# training for n number of epochs. We would then expect a run till n-times the total number of images in trainX (20300)
# For 10 epochs, the run (value of i) would be till 203000. And 2 models and 2 plots will be saved, one at 5th and the other at 10th epoch, as set.

dataset = [trainX, trainy]

# n_batch is images to be loaded everytime 

train(d_model, g_model, gan_model, dataset, n_epochs=20, n_batch=14) #2090) 


#_____________ALERT_________________# auditory indication of the end of execution (OPTIONAL)
from IPython.display import Audio
sound_file = "/Users/Dyutiman/Downloads/terminating alarm.mp3"
Audio(sound_file, autoplay=True)


# _________________________

# End of code!
