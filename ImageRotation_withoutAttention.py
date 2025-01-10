#!/usr/bin/env python
# coding: utf-8

# # Generative Network to rotate and obtain face-on PPD images
# ## SOURCE CODE
# ### Without Attention

#     This notebook is dedicated to training a Conditional Generative Adversarial Network (cGAN), incorporating the Pix2Pix concept, on a dataset of Protoplanetary Disk images obtained from the FARGO3D simulations. The primary goal of this project is to generate face-on images of Protoplanetary Disks from images that are rotated in random orientations.
# 
#     
#     Summary       : The aim of this notebook is to leverage the power of cGANs, particularly the Pix2Pix
#                     architecture, to transform images of Protoplanetary Disks with random orientations into 
#                     standardized face-on images. This transformation is crucial for various astronomical 
#                     studies and simulations where consistent orientation of disk images is required for 
#                     accurate analysis.
# 
# 
#     Code&Config   : The code is being done on Jupyter Notebook platform, and is being run on MacOS 13.1, 
#                     Apple M1, 8gb configuration.  
#                
#          
# ________________

# ### Supervisor   : Dr. Sayantan Auddy
# #### Written by    : Dyutiman Santra
# #### Updated       : 26th June, 2024
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

from math import ceil

#*****************

from sklearn.model_selection import train_test_split

#*****************

from numpy import asarray
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import randint

#*****************

import time
import sys

from datetime import datetime as dt
from IPython import display

#*****************


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

no_o_folders = 700        #to set the number of folders (total 700)
pxl = 256                #to set the desired pixel value
channel = 1             #to set channel, 1 for grayscale and 3 for coloured
save_interval = 20      #choosing the epoch interval to save model
num_epoch = 50
num_batch = 14

import ImageRotation_NetworkScript as NN

csv_directory = '/Users/Dyutiman/Documents/ML_Project/Pix2Pix/RT_Dataset_incl_posang.csv'
image_directory = '/Users/Dyutiman/Documents/ML_Project/Sayantan Da Projects/Final 1.5l'
# sound_file_directory = '/Users/Dyutiman/Downloads/terminating alarm.mp3'
saving_directory = '/Users/Dyutiman/Documents/ML_Project'

do_you_wish_to_resume_training = True
last_model = '/Users/Dyutiman/Documents/ML_Project/ClusterRun_Models/256/modelWeight_100_24-09-08_06_57.h5'


# ___________

# ||  Reading DATA csv ||

# In[4]:


df = pd.read_csv(csv_directory)  

# print(df,"\n")              #displaying csv

Ind, X_inlabel, Y_inlabel = [], [], []

Ind = df["index"]           #to store the image numbers excluding translational changes
X_label = df["incl"]        #to store the inclination angle
Y_label = df["posang"]      #to store the position angle

print("The dataframe is loaded.")


# ||  Reading and loading DATA images ||

# In[5]:


## Reading the Image Dataset, from specified folders

X = [] 															# a List to store oriented images
Y = [] 															# a List to store face-on image

k=0
m=0

print(f"Total number of folders to be loaded is {no_o_folders}.\n")

for i in range(1, no_o_folders+1):

    directory = image_directory +"/RT_A_"+ str(i)+"/*.png"
    data_set_indiv = glob.glob(directory)

    loc = image_directory +"/RT_A_"+ str(i)
    
    for j in Ind: #Loading the oriented images in X and correspondsing face-on images in Y
        X.append(loc+"/image_"+str(j)+".png")
        Y.append(loc+"/image_1.png")
        m = m+1
    
    k = k+1
    display.clear_output(wait=True)
    print("Total count of images = %d. ----> %2.2f %s"%(m,(k/no_o_folders)*100,'%'))

print(len(X), "image diretories are loaded.")


# ____

# In[6]:


# To display all the loaded images 
if (False):
    fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))

    k=0
    for i in X:
        try:
            axes.set_title("Run:{}".format(k))
            plt.imshow(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB))
            k=k+1
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.001)
            if(k%2 == 0):
                fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))
        except KeyboardInterrupt:
            break


# In[7]:


# deallocating the unreferenced objects and freeing up memory (OPTIONAL)
import gc
gc.collect()


# In[8]:


# Splitting the dataset
trainX, testX, trainy, testy = train_test_split(X, Y, random_state=42, test_size=0.10, shuffle=True)

# summarize the shape of the dataset
print('Train:', len(trainX), '\nTest:', len(testX), '\nTrainLabel:', len(trainy), '\nTestLabel:', len(testy))


# _________________________________

# # Preparing Training Block

# In[9]:


# selecting a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix//(len(Ind)-1)]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# In[10]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# _________________________________

# In[11]:


# loading a batch of images
def dataset_batch(lower, upper):
    TX = []
    TY = []
    p = upper

    for i in range(lower, upper):
        if(i==len(trainX)):
            p = i
            break
        image_dir = trainX[i]
        emag = cv2.imread(image_dir, channel//3)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TX.append(emag[58:428, 107:477])
        emag = cv2.flip(emag[58:428, 107:477],1)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TX.append(emag)
        
        trg_dir = trainy[i]
        emag = cv2.imread(trg_dir, channel//3)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TY.append(emag[58:428, 107:477])
        emag = cv2.flip(emag[58:428, 107:477],1)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TY.append(emag)
        
    print("Loading batch [%d --> %d] (%d)" % (lower,p,p-lower))
    
    tx = tf.image.resize(np.asarray(TX), [pxl, pxl])
    ty = tf.image.resize(np.asarray(TY), [pxl, pxl])

    T_X = tx.numpy()
    T_Y = ty.numpy()
    
    # scale from [0,255] to [-1,1]
    T_X = (T_X - 127.5) / 127.5
    T_Y = (T_Y - 127.5) / 127.5

    return [T_X,T_Y]


# In[12]:


# Generating samples and saving plot and the model 
def summarize_performance(epch, g_model, dataset, direct, n_samples=3):
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
        
    filename1 = 'plot_%02d_%s.png' % ((epch), FD)
    plt.savefig(f"{direct}/{filename1}")
    plt.close()
    # save the generator model
    filename2 = 'modelWeight_%02d_%s.h5' % ((epch), FD)
    g_model.save_weights(f"{direct}/{filename2}") 
    print('>Saved: %s and %s' % (filename1, filename2))


# In[13]:


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs, n_batch):
    
    start_time = time.time()
    last_time = start_time
    space = " "
    lowest_rec = 100
    highest_rec = 0
    
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    
    # calculate the number of batches per training epoch
    bat_per_epo = ceil(len(trainX) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    direct = ""      # to set up directory
    boolean = True   # flag variable to make sure that the directory is set only once
    
    counting_batch_per_epoch = 1
    dataset_index = 0
    
    # manually enumerate epochs
    for i in range(n_steps):
        
        #_____________________________________________________________________________Training Summary
        print(f"{space*65}______________________________________\n")
        print(f"{space*65}Epoch                    : {i//bat_per_epo +1} / {n_epochs}")
        print(f"{space*65}Steps per epoch          : {counting_batch_per_epoch} / {bat_per_epo}")
        #_____________________________________________________________________________Training Summary
        
        #taking batch
        dataset = dataset_batch(dataset_index, dataset_index+n_batch)
        dataset_index = dataset_index+n_batch
        
        print("Total loaded images", len(dataset[0]))
        
        
        if(counting_batch_per_epoch==bat_per_epo):
            dataset_index = 0
            counting_batch_per_epoch = 0
            
        
        # selecting a batch of real samples
        [X_realA, X_realB], y_real = dataset, ones((len(dataset[0]), n_patch, n_patch, 1)) #generate_real_samples(dataset, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        

        # updating discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        
        # summarizing performance
        display.clear_output(wait=True) # ensures a single line print instead of multiple lines (OPTIONAL)
        print(('>%d, d1[%.3f] d2[%.3f] g[%.3f] --->%2.2f %s ' % (i+1, d_loss1, d_loss2, g_loss, (i+1)/n_steps*100, '%')))
        
        
        #_____________________________________________________________________________Training Summary
        current_time = time.time()
        current_time_elapsed = current_time - start_time
        time_left = (current_time - last_time)*(n_steps-i-1)
        print(f"{space*65}Time Elaspsed{space*12}: {int(current_time_elapsed//3600)}h {int((current_time_elapsed%3600)//60)}m {int(current_time_elapsed%60)}s\n{space*65}Approximate Time Left{space*4}: {int(time_left//3600)}h {int((time_left%3600)//60)}m {int(time_left%60)}s")
        
        
        if((current_time - last_time)>highest_rec): highest_rec = current_time - last_time
        if((current_time - last_time)<lowest_rec): lowest_rec = current_time - last_time
        
        print(f"{space*65}______________________________________\n")
        print(f"{space*65}\033[1mTIME PER STEP\033[0m")
        print(f"%sLowest recorded          : %.2fs"%(space*65,lowest_rec))
        print(f"%sCurrent time          ---> \033[1m%.2fs\033[0m"%(space*65,current_time - last_time))
        print(f"%sHighest recorded         : %.2fs"%(space*65,highest_rec))
        
        
        last_time = current_time
        #_____________________________________________________________________________Training Summary
        
        
        # summarizing model performance
        if (i+1) % (bat_per_epo * save_interval) == 0: # saving model and plot at every 5th epoch
            print('Summarizing and Saving')
            if(boolean):
                print("Setting up directory...")
                formatted_date = (dt.now()).strftime("%Y-%m-%d %H:%M:%S")
                FD = "ModelWeight_"+formatted_date[2:10]+"_"+formatted_date[11:13]+"_"+formatted_date[14:16]
                direct = saving_directory+"/Pix2Pix/%s"%(FD)
                os.mkdir(direct) #creating directory
                boolean = False
            summarize_performance(i//bat_per_epo +1, g_model, dataset, direct)
            gc.collect() #freeing up memory (OPTIONAL)
            
        counting_batch_per_epoch=counting_batch_per_epoch+1
            
            
            
# #_____________ALERT_________________# auditory indication of the end of execution (OPTIONAL)
#     from IPython.display import Audio
#     sound_file = "/Users/Dyutiman/Downloads/terminating alarm.mp3"
#     Audio(sound_file, autoplay=True)
    


# In[14]:


# defining input shape based on the loaded dataset
image_shape = (pxl,pxl,channel)
print("Shape is",image_shape)

# define the models
print("Working so far : d")
d_model = NN.define_discriminator(image_shape)

print("Working so far : g")
g_model = NN.define_generator(image_shape)

# define the composite model
gan_model = NN.define_gan(g_model, d_model, image_shape)


# In[15]:


if(do_you_wish_to_resume_training):
    latest_weights = max(glob.glob(last_model), key=os.path.getmtime)
    print(latest_weights)
    g_model.load_weights(latest_weights)


# ### Training Block preparation complete.

# __________________________________________________________________

# # TRAINING !

# In[16]:


dataset = [trainX, trainy]

# n_batch is the number images to be loaded everytime (the number is doubled as mirror-images are also generated)

train(d_model, g_model, gan_model, dataset, n_epochs=num_epoch, n_batch=num_batch)


# #_____________ALERT_________________# auditory indication of the end of execution (OPTIONAL)
# from IPython.display import Audio
# sound_file = sound_file_directory
# Audio(sound_file, autoplay=True)


# ___

# End of code!
