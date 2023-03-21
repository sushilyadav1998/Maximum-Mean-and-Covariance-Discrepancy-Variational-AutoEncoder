# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:54:21 2020

@author: Sushilkumar.Yadav
"""
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.models import load_model
from keras.layers import Lambda, Input, Dense, Conv2D
from keras.models import Model
from keras.optimizers import adam
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import cv2
from random import shuffle
from tqdm import tqdm
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import models
import matplotlib.cm as cm
from keras.layers import Reshape, Conv2DTranspose
from keras.losses import mse
from collections import Counter
import joblib

#%%

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#%%
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
#%%
def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    print(z_mean.shape)
    plt.figure(figsize=(12,11))
    #dict = {ms[i]: markers[i] for i in ms}
    mscatter(z_mean[:, 0], z_mean[:, 1], c=y_test, m=test2, s=100, edgecolors='black')
        
    #%%
    ys = np.unique(y_test)
    means = np.array([np.mean(z_mean[y_test == y, :], axis=0) for y in ys])
    plt.scatter(means[:, 0], means[:, 1], c=ys, s=200, edgecolors='black')

    #%%
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 1
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-10, 10, n)
    grid_y = np.linspace(-10, 10, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            #%%
            #print(x_decoded.shape)
            #%%
            digit = x_decoded[0].reshape(digit_size, digit_size)
            #cv2.imwrite(r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\standard_database\test_images\test.'+ str(i)+'.jpeg',255*digit)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 11))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

#%%
data = ["Aishwarya_Rai", "Alia_Bhatt", "Amitabh_Bachchan", "Disha_Patani", "Hrithik_Roshan", "Jacqueline_Fernandez", "Salman_Khan", "Shah_Rukh_Khan", "Shahid_Kapoor", "Shraddha_Kapoor", "Varun_Dhawan"]
d1=r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\standard_database\test_images'
Img_size=64
def create_training_data(d1):
    training_data=[]
    for i in data:
        Train_dir = d1
        print(Train_dir)
        label=2
        for img in tqdm(os.listdir(Train_dir)):
            path = os.path.join(Train_dir,img)
        #print(path,label)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
            training_data.append([np.array(img),np.array(label)])
        #shuffle(training_data)
    #np.save('sc_train_data.npy',training_data)
    return training_data
X=create_training_data(d1)
shuffle(X)

x_i = np.array([i[0] for i in X]).reshape(-1, Img_size, Img_size, 1)
y_i = np.array([i[1] for i in X])
x_train=x_i
y_train=y_i
x_test=x_train
y_test=y_train
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#%%
    
image_size = x_train.shape[1]
original_dim = image_size * image_size
input_shape = (original_dim, )
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# compute the number of labels
num_labels = len(np.unique(y_train))


#%%

test1 = []
test_shape = y_test.shape
markers = ['.','o','v','^','>','<','s','p','*','h','H']
def convert_to_1d():
    
    #testvalue=[]
    for i in range(test_shape[0]):
        if (y_test[i] == 0):
            test1.append(markers[0])
        elif (y_test[i] == 1):
            test1.append(markers[1])
        elif (y_test[i] == 2):
            test1.append(markers[2])
        elif (y_test[i] == 3):
            test1.append(markers[3])
        elif (y_test[i] == 4):
            test1.append(markers[4])
        elif (y_test[i] == 5):
            test1.append(markers[5])
        elif (y_test[i] == 6):
            test1.append(markers[6])
        elif (y_test[i] == 7):
            test1.append(markers[7])
        elif (y_test[i] == 8):
            test1.append(markers[8])
        elif (y_test[i] == 9):
            test1.append(markers[9])
        elif (y_test[i] == 10):
            test1.append(markers[10])
            
    return test1
        
test2 = convert_to_1d()

#%%
# network parameters

intermediate_dim = 3600
batch_size = 64
latent_dim = 2
epochs = 10 #default:50

# VAE model = encoder + decoder
# build encoder model

#%%
inputs = Input(shape=input_shape, name='encoder_input')

#x = inputs
#
#x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
#x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
#
#shape = K.int_shape(x)
#%%
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

#x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
#x = Reshape((shape[1], shape[2], shape[3]))(x)
#
#x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
#x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
#
#outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)
#

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    #opt = adam(lr=1e-3, decay=1e-9)
    vae.compile(optimizer='adam')
    vae.load_weights('weightsmmcd2000epoch.h5')
    
#%%
    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
    
#%%
#encoder, decoder = models  
#print(x_test[0].shape)
##%% 
#x_encoded = encoder.predict(x_test[0])
#print(x_encoded.shape)
##%%
#x_decoded = decoder.predict(x_encoded)
#
#print(x_encoded.shape, x_decoded.shape)

#%%
#n = 50
#for i in range(n):
#    decoded = x_decoded[i].reshape(64,64)
#    #cv2.imshow('output',decoded)
#    cv2.waitKey(0)
#    cv2.imwrite('Tony1/Tony_Blair.'+ str(i)+'.jpeg',255*decoded)
#    #fig.savefig('face'+ str(i)+ '.jpg') use this to save the fig when plt is used
#    
#save_img()    
#plt.show()
#    
    
    