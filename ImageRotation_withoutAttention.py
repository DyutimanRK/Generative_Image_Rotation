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
import csv
import re

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


# Constants and default settings
no_o_folders = 700                               # Total number of folders
pxl = 256                                        # Desired pixel value
save_interval = 20                               # Epoch interval to save the model
num_epoch = 200                                  # Number of epochs

channel = 1                                      # 1 for grayscale, 3 for colored
num_batch = 14                                   # Batch size
do_you_wish_to_resume_training = True


# _____________________________________________________________________________________________________________

import ImageRotation_NetworkScript as NN

# Base directory setup
base_dir = os.getcwd()
saving_dir = os.path.join(base_dir, "Models") # Create a saving directory in the base folder

# Dynamically locate required files and directories
def locate_file(file_pattern, search_dir=base_dir):
    """Search for a specific file in a given directory or subdirectories."""
    files = glob.glob(os.path.join(search_dir, file_pattern), recursive=True)
    if files:
        return files[0]  # Return the first match
    else:
        print(f"File not found: {file_pattern}")
        return None

# Locate directories and files
csv_directory = locate_file("RT_Dataset_incl_posang.csv")
image_directory = locate_file("Disk_gas_plots")

    
# Function to get the latest folder based on timestamp that starts without "ATTN"
def get_latest_folder(directory):
    """Returns the folder starting without 'ATTN' with the most recent timestamp."""
    try:
        # Filter folders that start without "ATTN"
        folders = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f)) and f.startswith("NoATTN")
        ]
        # Sort by modification time (latest first)
        folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return folders[0] if folders else None
    except Exception as e:
        print(f"Error finding latest folder in {directory}: {e}")
        return None


# Locate the latest model folder if resuming training
if do_you_wish_to_resume_training:
    latest_model_folder = get_latest_folder(saving_dir)
    if latest_model_folder:
        print(f"Latest model folder: {latest_model_folder}")
        # Search for the latest model (h5) file in the latest folder
        model_files = glob.glob(os.path.join(latest_model_folder, "*weights.h5"))
        if model_files:
            # Sort model files by timestamp (modification time) and pick the latest
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            last_model = model_files[0]
            print(f"Resuming training with model: {last_model}")
        else:
            print(f"No model found in {latest_model_folder}. Starting fresh training.")
            last_model = None
    else:
        print("No previous model folder found. Starting fresh training.")
        last_model = None
else:
    last_model = None

last_model_epoch = 0
if last_model:
    last_model_epoch = int(re.search(r'EP(\d+)_', os.path.basename(last_model)).group(1))

# Output paths for verification
print(f"Base Directory: {base_dir}")
print(f"CSV Directory: {csv_directory}")
print(f"Image Directory: {image_directory}")
print(f"Saving Directory: {saving_dir}")
print(f"Last Model: {last_model}")


# In[4]:


# Create a folder named "Readings" to store the CSV files
readings_dir = os.path.join(os.getcwd(), "Readings")  # Folder path
os.makedirs(readings_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Set up CSV logging
if do_you_wish_to_resume_training:
    csv_file = os.path.join(readings_dir, f"{pxl}_NoATTN_Training_Log.csv")
else:
    formatted_date = (dt.now()).strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = os.path.join(readings_dir, f"{pxl}_NoATTN_Training_Log_{formatted_date}.csv")

csv_headers = ["Date", "Epoch", "Cumulative Epoch", "Training G Loss", "Validation G Loss"]

# Ensure the file exists and add the header only once
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)  # Write the header row


# ___________

# ||  Reading DATA csv ||

# In[5]:


df = pd.read_csv(csv_directory)  

# print(df,"\n")              #displaying csv

Ind = df["index"]           #to store the image numbers excluding translational changes
X_label = df["incl"]        #to store the inclination angle
Y_label = df["posang"]      #to store the position angle

print("The dataframe is loaded.")


# ||  Reading and loading DATA images ||

# In[6]:


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

# In[7]:


# # To display all the loaded images 
# if (False):
#     fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))

#     k=0
#     for i in X:
#         try:
#             axes.set_title("Run:{}".format(k))
#             plt.imshow(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB))
#             k=k+1
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#             time.sleep(0.001)
#             if(k%2 == 0):
#                 fig, axes = plt.subplots(ncols=1, sharex=False,sharey=True, figsize=(15, 5))
#         except KeyboardInterrupt:
#             break


# In[8]:


# # deallocating the unreferenced objects and freeing up memory (OPTIONAL)
# import gc
# gc.collect()


# In[9]:


# Splitting the dataset
trainX, testX, trainy, testy = train_test_split(X, Y, random_state=42, test_size=0.10, shuffle=True)
trainX, testX, trainy, testy = train_test_split(trainX, trainy, random_state=42, test_size=0.15, shuffle=True)

# summarize the shape of the dataset
print('Train:', len(trainX), '\nTest:', len(testX), '\nTrainLabel:', len(trainy), '\nTestLabel:', len(testy))


# _________________________________

# # Preparing Training Block

# In[10]:


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


# In[11]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# _________________________________

# In[12]:


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


# In[13]:


# loading a batch of images
def val_dataset_batch(lower, upper):
    TX = []
    TY = []
    p = upper

    for i in range(lower, upper):
        if(i==len(testX)):
            p = i
            break
        image_dir = testX[i]
        emag = cv2.imread(image_dir, channel//3)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TX.append(emag[58:428, 107:477])
        emag = cv2.flip(emag[58:428, 107:477],1)
        if(channel==1): emag = np.expand_dims(emag, axis=-1)
        TX.append(emag)
        
        trg_dir = testy[i]
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


# In[14]:


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
    formatted_date = dt.now().strftime("%Y-%m-%d %H-%M-%S")
    FD = formatted_date[2:10]+"_"+formatted_date[11:13]+"_"+formatted_date[14:16]
        
    # Save plot to file
    filename1 = f'NoATTN_plot_EP{epch:03d}_{FD}.png'
    plot_path = os.path.join(direct, filename1)
    plt.savefig(plot_path)
    plt.close()

    # Save the generator model
    filename2 = f'NoATTN_modelWeight_EP{epch:03d}_{FD}.weights.h5'
    model_path = os.path.join(direct, filename2)
    g_model.save_weights(model_path)
     
    print(f'>Saved: {filename1} and {filename2}')


# In[15]:


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, val_batch, n_epochs, n_batch):
    
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
        print(f"{space*65}Time Elapsed{space*12}: {int(current_time_elapsed//3600)}h {int((current_time_elapsed%3600)//60)}m {int(current_time_elapsed%60)}s\n{space*65}Approximate Time Left{space*4}: {int(time_left//3600)}h {int((time_left%3600)//60)}m {int(time_left%60)}s")
        
        
        if((current_time - last_time)>highest_rec): highest_rec = current_time - last_time
        if((current_time - last_time)<lowest_rec): lowest_rec = current_time - last_time
        
        print(f"{space*65}______________________________________\n")
        print(f"{space*65}\033[1mTIME PER STEP\033[0m")
        print(f"%sLowest recorded          : %.2fs"%(space*65,lowest_rec))
        print(f"%sCurrent time          ---> \033[1m%.2fs\033[0m"%(space*65,current_time - last_time))
        print(f"%sHighest recorded         : %.2fs"%(space*65,highest_rec))
        
        
        last_time = current_time
        #_____________________________________________________________________________Training Summary
        
        
        # Validation Loss Calculation at end of each epoch
        if (i + 1) % (5*bat_per_epo) == 0:  # end of each epoch
            [val_realA, val_realB] = val_batch            
            val_fakeB = g_model.predict(val_realA)
            val_d_loss_real = d_model.evaluate([val_realA, val_realB], ones((len(val_realA), n_patch, n_patch, 1)), verbose=0)
            val_d_loss_fake = d_model.evaluate([val_realA, val_fakeB], zeros((len(val_realA), n_patch, n_patch, 1)), verbose=0)
            val_g_loss, _, _ = gan_model.evaluate(val_realA, [ones((len(val_realA), n_patch, n_patch, 1)), val_realB], verbose=0)
            
            print(f"\nValidation Loss - Epoch {i//bat_per_epo + 1}: d1_val[{val_d_loss_real}] d2_val[{val_d_loss_fake}] g_val[{val_g_loss}]")

            
            # Log to CSV
            epoch = i // bat_per_epo + 1
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([(dt.now()).strftime("%Y-%m-%d %H-%M-%S"), epoch, last_model_epoch+epoch, g_loss, val_g_loss])
                
        
        # Summarizing model performance
        if (i + 1) % (bat_per_epo * save_interval) == 0:  # saving model and plot at every save_interval epoch
            print('Summarizing and Saving')
            if (boolean):
                if last_model:  # If the model is loaded
                    print("Using the existing folder for saving...")
                    # Extract the folder path from the last_model
                    direct = os.path.dirname(last_model)

                    # Update the folder name with the new date
                    formatted_date = (dt.now()).strftime("%Y-%m-%d %H-%M-%S")
                    new_folder_name = f"NoATTN_ModelWeight_{formatted_date[2:10]}_{formatted_date[11:13]}_{formatted_date[14:16]}"
                    new_direct = os.path.join(saving_dir, new_folder_name)

                    # Rename the folder
                    os.rename(direct, new_direct)
                    direct = new_direct  # Use the renamed folder
                else:
                    print("Setting up a new directory...")
                    formatted_date = (dt.now()).strftime("%Y-%m-%d %H-%M-%S")
                    FD = f"NoATTN_ModelWeight_{formatted_date[2:10]}_{formatted_date[11:13]}_{formatted_date[14:16]}"

                    # Create a specific directory for this run
                    direct = os.path.join(saving_dir, FD)
                    os.makedirs(direct, exist_ok=True)  # Create the directory

                boolean = False  # Prevent directory setup from happening again
                
            summarize_performance(last_model_epoch+(i//bat_per_epo +1), g_model, dataset, direct)
#             gc.collect() #freeing up memory (OPTIONAL)
            
        counting_batch_per_epoch=counting_batch_per_epoch+1


# In[16]:


# Define input shape based on the loaded dataset
image_shape = (pxl, pxl, channel)
print("Shape is", image_shape)

# Define the models
print("Working so far: d")
d_model = NN.define_discriminator(image_shape)

print("Working so far: g")
g_model = NN.define_generator(image_shape)

# Load weights into g_model if a last_model is provided
if last_model is not None:
    print(f"Loading weights from {last_model} into g_model...")
    g_model.load_weights(last_model)
else:
    print("No previous model found. Using a fresh g_model.")

# Define the composite model
gan_model = NN.define_gan(g_model, d_model, image_shape)


# ### Training Block preparation complete.

# __________________________________________________________________

# # TRAINING !

# In[17]:


dataset = [trainX, trainy]
val_dataset = [testX, testy]
val_batch = val_dataset_batch(0, len(val_dataset[0]))

# n_batch is the number images to be loaded everytime (the number is doubled as mirror-images are also generated)

train(d_model, g_model, gan_model, dataset, val_batch, n_epochs=num_epoch, n_batch=num_batch)


# ___

# End of code!
