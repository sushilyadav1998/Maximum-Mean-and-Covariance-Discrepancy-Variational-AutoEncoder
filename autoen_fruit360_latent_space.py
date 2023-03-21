# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:04:12 2020

@author: Sushilkumar.Yadav
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models,layers
from keras import applications
import glob2 as glob
from numpy import random
from tqdm import tqdm
import os
import cv2
from random import shuffle

#%%
# dimensionality of the latents space 
embedding_dim = 3600

#Input layer
input_img = layers.Input(shape=(4096,))  

#Encoding layer
encoded = layers.Dense(embedding_dim, activation='relu')(input_img)

#Decoding layer
decoded = layers.Dense(4096,activation='sigmoid')(encoded) 

#Autoencoder --> in this API Model, we define the Input tensor and the output layer
#wraps the 2 layers of Encoder e Decoder
autoencoder = models.Model(input_img,decoded)
autoencoder.summary()

#Encoder
encoder = models.Model(input_img,encoded)
encoder.summary()

#Decoder
encoded_input = layers.Input(shape=(embedding_dim,))
decoder_layers = autoencoder.layers[-1]  #applying the last layer
decoder = models.Model(encoded_input,decoder_layers(encoded_input))

print(input_img)
print(encoded)
#%%
autoencoder.compile(
    optimizer='adadelta',  #backpropagation Gradient Descent
    loss='binary_crossentropy'
)
#%%
fruits = ["Apricot", "Avocado", "Banana", "Cocos", "Fig", "Guava", "Kiwi", "Plum", "Pomegranate", "Raspberry", "Strawberry"]
d1=r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\standard_database\fruits-360\Training'
Img_size=64
def create_training_data(d1):
    training_data=[]
    for i in fruits:
        Train_dir = d1+'\\'+i+'\\'
        print(Train_dir)
        label=fruits.index(i)
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


x_train = x_train.reshape((-1,4096))  #to go from (60000,28,28) to new shape and -1 let
                                    #numpy to calculate the number for you
x_test = x_train.reshape((-1,4096))

print(x_train.shape,x_test.shape)

#%%

history = autoencoder.fit(x_train,x_train,epochs=3,batch_size=512,shuffle=True,
                validation_data=(x_test,x_test))
                
#%%
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
#%%


encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs)  
print(encoded_imgs.shape,decoded_imgs.shape)
    
#%%

n = 50
#plt.figure(figsize=(20,4))
for i in range(n):
#    ax = plt.subplot(2, n, i+1)
#    plt.imshow(x_test[i].reshape((64,64)),cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    
#    ax = plt.subplot(2,n,i+1+n)
#    plt.imshow(decoded_imgs[i].reshape((64,64)),cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
    decoded=decoded_imgs[i].reshape((64,64))
    #cv2.imshow('output',decoded)
    #cv2.waitKey(0)
    #cv2.imwrite('AE_XRay/xray.'+ str(i)+'.jpeg',255*decoded)
    #fig.savefig('face'+ str(i)+ '.jpg')
    plt.show()
    
#%%
    
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
fig, ax = plt.subplots()
mscatter(encoded_imgs[:,0], encoded_imgs[:,1], c=y_test, m=test2, s=100, edgecolors='black')
ys = np.unique(y_test)
means = np.array([np.mean(encoded_imgs[y_test == y, :], axis=0) for y in ys])
plt.scatter(means[:, 0], means[:, 1], c=ys, s=200, edgecolors='black')
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()