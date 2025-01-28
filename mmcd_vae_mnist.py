# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:39:20 2020

@author: Sushilkumar.Yadav
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
import os
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from tensorflow import keras
from keras.datasets import mnist

#%%
#parent_dir = ["bollywood_celeb_faces_0", "bollywood_celeb_faces_1", "bollywood_celeb_faces2"]
INPUT_DIM = (64,64,3) # Image dimension
BATCH_SIZE = 64
Img_size = 64
Z_DIM = 200

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%
no_of_images = x_train.shape
no_of_images = no_of_images[0]
image_size = x_train.shape[1]
#original_dim = image_size * image_size
x_train = x_train[-1]
x_test = x_test[-1]

#%%
x_train = cv2.cvtColor(x_train,cv2.COLOR_GRAY2RGB)
x_test = cv2.cvtColor(x_test,cv2.COLOR_GRAY2RGB)

#%%
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

#%%
# ENCODER
def vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, 
                  conv_strides, use_batch_norm = False, use_dropout = False):
  
  # Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu, 
  # BatchNormalization and Dropout.
  # Otherwise, the names of above mentioned layers in the model 
  # would be inconsistent
  global K
  K.clear_session()
  
  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  encoder_input = Input(shape = input_dim, name = 'encoder_input')
  x = encoder_input

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2D(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'encoder_conv_' + str(i)
                  )(x)
      if use_batch_norm:
        x = BatchNormalization()(x)
  
      x = LeakyReLU()(x)

      if use_dropout:
        x = Dropout(rate=0.25)(x)

  # Required for reshaping latent vector while building Decoder
  shape_before_flattening = K.int_shape(x)[1:] 
  
  x = Flatten()(x)
  
  z_mean = Dense(output_dim, name = 'mu')(x)
  z_log_var = Dense(output_dim, name = 'z_log_var')(x)

  # Defining a function for sampling
  def sampling(args):
    z_mean, z_z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.) 
    return z_mean + K.exp(z_z_log_var/2)*epsilon   
  
  # Using a Keras Lambda Layer to include the sampling function as a layer 
  # in the model
  encoder_output = Lambda(sampling, name='encoder_output')([z_mean, z_log_var])

  return encoder_input, encoder_output, z_mean, z_log_var, shape_before_flattening, Model(encoder_input, encoder_output)
#%%
vae_encoder_input, vae_encoder_output,  z_mean, z_log_var, vae_shape_before_flattening, vae_encoder  = vae_encoder(input_dim = INPUT_DIM,
                                    output_dim = Z_DIM, 
                                    conv_filters = [32, 64, 64, 64],
                                    conv_kernel_size = [3,3,3,3],
                                    conv_strides = [2,2,2,2])
vae_encoder.summary()
#%%
# Decoder
def vae_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, 
                  conv_strides):

  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  vae_decoder_input = Input(shape = (input_dim,) , name = 'vae_decoder_input')

  # To get an exact mirror image of the encoder
  x = Dense(np.prod(shape_before_flattening))(vae_decoder_input)
  x = Reshape(shape_before_flattening)(x)

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2DTranspose(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'decoder_conv_' + str(i)
                  )(x)
      
      # Adding a sigmoid layer at the end to restrict the outputs 
      # between 0 and 1
      if i < n_layers - 1:
        x = LeakyReLU()(x)
      else:
        x = Activation('sigmoid')(x)

  # Define model output
  vae_decoder_output = x

  return vae_decoder_input, vae_decoder_output, Model(vae_decoder_input, vae_decoder_output)

#%%
vae_decoder_input, vae_decoder_output, vae_decoder = vae_decoder(input_dim = Z_DIM,
                                        shape_before_flattening = vae_shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [2,2,2,2]
                                        )
vae_decoder.summary()

#%%
# The input to the model will be the image fed to the encoder.
vae_input = vae_encoder_input

# Output will be the output of the decoder. The term - decoder(encoder_output) 
# combines the model by passing the encoder output to the input of the decoder.
vae_output = vae_decoder(vae_encoder_output)

# Input to the combined model will be the input to the encoder.
# Output of the combined model will be the output of the decoder.
vae_model = Model(vae_input, vae_output)

vae_model.summary()

#%%
LEARNING_RATE = 0.0004
N_EPOCHS = 10000
LOSS_FACTOR = 10000

#%%
def kl_loss(y_true, y_pred):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

def reconstruction_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile   (tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def compute_mcd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    yx_kernel = compute_kernel(y, x)
    squared_x_kernel = tf.square(x_kernel)
    squared_y_kernel = tf.square(y_kernel)
    squared_xy_kernel = tf.square(xy_kernel)
    #squared_yx_kernel = tf.square(yx_kernel)
    return (tf.reduce_mean(squared_x_kernel) - 2 * tf.reduce_mean(tf.square(tf.reduce_mean(x_kernel)))
                + tf.square(tf.reduce_mean(x_kernel)) - 2 * tf.reduce_mean(squared_xy_kernel)
                + 2 * tf.reduce_mean(tf.square(tf.reduce_mean(xy_kernel)))
                + 2 * tf.reduce_mean(tf.square(tf.reduce_mean(yx_kernel)))
                - 2 * tf.square(tf.reduce_mean(xy_kernel)) + tf.reduce_mean(squared_y_kernel)
                - 2 * tf.reduce_mean(tf.square(tf.reduce_mean(y_kernel)))
                + tf.square(tf.reduce_mean(y_kernel)))

#%%
true_samples = tf.random_normal(tf.stack([200, Z_DIM]))
loss_mmd = compute_mmd(true_samples, vae_encoder_output)
print(loss_mmd)
loss_mcd = compute_mcd(true_samples, vae_encoder_output)
print(loss_mcd)
beta = 0.01
loss_mmcd = loss_mmd + beta * loss_mcd
print(loss_mmcd)
def total_loss(y_true, y_pred):
    return LOSS_FACTOR*reconstruction_loss(y_true, y_pred) + loss_mmcd
    #return LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)
    #return LOSS_FACTOR*reconstruction_loss(y_true, y_pred) + loss_mmd
#%%

adam_optimizer = Adam(lr = LEARNING_RATE)

vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [reconstruction_loss])

checkpoint_vae = ModelCheckpoint(os.path.join('VAE_Result_Working/weightsmmcd10000epoch.h5'), save_weights_only = True, verbose=1)

#%%
vae_model.fit(x_train, 
                        shuffle=True, 
                        epochs = N_EPOCHS, 
                        initial_epoch = 0, 
                        steps_per_epoch=no_of_images / BATCH_SIZE,
                        callbacks=[checkpoint_vae])


#%%
#example_batch = next(data_flow)
#example_batch = example_batch[0]
#example_images = example_batch[:20]
example_images = x_test[:20]

#%%

def plot_compare_vae(images=None, add_noise=False):
  
  if images is None:
#    example_batch = next(data_flow)
#    example_batch = example_batch[0]
    images = x_test[:10]

  n_to_show = images.shape[0]
  #reconst_images = vae_model.predict(images)

  if add_noise:
      encodings = vae_encoder.predict(images)
      encodings += np.random.normal(0.0, 1.0, size = (n_to_show,200))
      reconst_images = vae_decoder.predict(encodings)

  else:
      reconst_images = vae_model.predict(images)


  fig = plt.figure(figsize=(15, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)
  for i in range(n_to_show):
      img = images[i].squeeze()
      sub = fig.add_subplot(2, n_to_show, i+1)
      sub.axis('off')        
      sub.imshow(img)
      cv2.imwrite(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\VAE_Result_Working\10000epmmcdtest1."+ str(i)+".jpeg", 255*img)

  for i in range(n_to_show):
      img = reconst_images[i].squeeze()
      sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
      sub.axis('off')
      sub.imshow(img)
      cv2.imwrite(r"C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\VAE_Result_Working\10000epmmcdvaegen."+ str(i)+".jpeg", 255*img)
      
#%%
plot_compare_vae(images = example_images)

#%%
def vae_generate_images(n_to_show=10):
  reconst_images = vae_decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_DIM)))

  fig = plt.figure(figsize=(15, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)

  for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)
        
#%%
vae_generate_images(n_to_show=10)

#%%
#labels = (data_flow.class_indices)
#labels_dict = dict((v,k) for k,v in labels.items())

#predictions = [labels[k] for k in predicted_class_indices]
#%%
#code for Normalization plot
from scipy.stats import norm
z_test = vae_encoder.predict(x_test[:20])

x = np.linspace(-3, 3, 300)

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(50):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[:,i], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))

plt.show()
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
color_map = [color for color in labels.values()]
plt.figure()
z_mean = vae_encoder.predict(example_images)
mscatter(z_mean[:, 0], z_mean[:, 1],c = y_test, s=100, edgecolors='black')
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()
