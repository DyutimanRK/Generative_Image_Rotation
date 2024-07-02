#!/usr/bin/env python
# coding: utf-8

# # Generative Network to rotate and obtain face-on PPD images
# ## SOURCE CODE
# ### Network Script

#     This script is dedicated to defining the architecture of a Conditional Generative Adversarial Network (cGAN), specifically implementing the Pix2Pix model. The network is designed to be trained on a dataset of Protoplanetary Disk images obtained from the FARGO3D simulations. The primary goal of the network is to generate face-on images of Protoplanetary Disks from images that are rotated in random orientations.
# 
#     
#     Summary       : The purpose of this script is to establish the network architecture for leveraging the 
#                     power of cGANs, particularly the Pix2Pix framework, to transform images of Protoplanetary 
#                     Disks with random orientations into standardized face-on images. This architectural design
#                     is crucial for various astronomical studies and simulations, where consistent orientation 
#                     of disk images is required for accurate analysis.
# 
# 
#     Code&Config   : The code is being done on Jupyter Notebook platform, and is being run on MacOS 13.1, 
#                     Apple M1, 8gb configuration.  
#                
#          
# ________________

# ### Supervisor   : Dr. Sayantan Auddy
# #### Written by    : Dyutiman Santra
# #### Created       : 24th June, 2024
# 
# _______________________________________

# ||  Importing Modules  ||

# In[1]:


from tensorflow.keras import layers, losses

from keras import backend as K
from keras.layers import BatchNormalization, Activation, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape, Dropout, LeakyReLU, Embedding, Concatenate, Multiply, Add
from keras.models import Model
from keras.losses import binary_crossentropy

from keras.initializers import RandomNormal

from keras.optimizers import Adam

#*****************

from numpy import ones
from numpy import zeros
from numpy.random import randint

#*****************


# _________________________________

# ## Network Setup

# #### Discriminator, Encoder, Decoder, GAN

# In[2]:


def define_discriminator(image_shape):
    
    # weight initialization
    init = RandomNormal(stddev=0.02) 
    
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

########################################################################################################################

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

########################################################################################################################

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

########################################################################################################################

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
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model


# ______________________________

# ## Attention Mechanism

# In[3]:


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

########################################################################################################################

# Define decoder block with attention
def decoder_block_with_attention(layer_in, skip_in, n_filters, dropout=True):
    # Decoder block as before
    g = decoder_block(layer_in, skip_in, n_filters, dropout=dropout)
    # Apply attention mechanism
    attention = attention_gate(g, skip_in, n_filters // 2)
    # Concatenate attention and decoder block output
    g = Concatenate()([g, attention])
    return g

########################################################################################################################

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


# _____________________________

# ### Generator without Attention

# In[4]:


# define the standalone generator model - U-net
def define_generator(image_shape):      
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)

    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
#         e6 = define_encoder_block(e5, 512)
#         e7 = define_encoder_block(e6, 512) #this should be included if images are 256x256 (e6, e7, b,d1, d2 should be accordingly placed in places)
#         e8 = define_encoder_block(e7, 512) #this should be included if images are 512x512
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5) #this would be (e7) in case of 256x256
    b = Activation('relu')(b)

#         d0 = decoder_block(b, e8, 512) #this should be included if images are 512x512
#         d1 = decoder_block(b, e7, 512) #this should be included if images are 256x256
#         d2 = decoder_block(d1, e6, 512) #the b will change to d1 if resolution is increased
    d3 = decoder_block(b, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7) #Modified 
    out_image = Activation('tanh')(g)  #Generates images in the range -1 to 1. So we change inputs also to -1 to 1

    # define model
    model = Model(in_image, out_image)
    return model


# ### Generator with Attention

# In[5]:


# Define the standalone generator model - U-net with attention
def define_generator_with_attention(image_shape):

    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)

    e1 = define_encoder_block_with_attention(in_image, 64, batchnorm=False)
    e2 = define_encoder_block_with_attention(e1, 128)
    e3 = define_encoder_block_with_attention(e2, 256)
    e4 = define_encoder_block_with_attention(e3, 512)
    e5 = define_encoder_block_with_attention(e4, 512)
#         e6 = define_encoder_block_with_attention(e5, 512)
#         e7 = define_encoder_block_with_attention(e6, 512)
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
#         d1 = decoder_block_with_attention(b, e7, 512)
#         d2 = decoder_block_with_attention(d1, e6, 512)
    d3 = decoder_block_with_attention(b, e5, 512)
    d4 = decoder_block_with_attention(d3, e4, 512, dropout=False)
    d5 = decoder_block_with_attention(d4, e3, 256, dropout=False)
    d6 = decoder_block_with_attention(d5, e2, 128, dropout=False)
    d7 = decoder_block_with_attention(d6, e1, 64, dropout=False)

    g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


# ### Network Setup complete.

# _________________________________
