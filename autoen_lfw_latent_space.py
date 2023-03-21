# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:58:22 2020

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
Train_dir = r'C:\Users\sushilkumar.yadav\Desktop\vmware\Personal\Research\Image_recognition_in_wild_using_Deep_Learning\Database_FR\LFWdataset1'
Img_size = 64 #256

LR = 0.01
def label_img(img):
    word_label = img.split('.')[0]
    if word_label =='Ariel_Sharon':
        return [1,0,0,0,0,0,0,0,0,0,0]
    elif word_label =='Colin_Powell':
        return [0,1,0,0,0,0,0,0,0,0,0]
    elif word_label =='George_Bush':
        return [0,0,1,0,0,0,0,0,0,0,0]
    elif word_label =='Gerhard_Schroed':
        return [0,0,0,1,0,0,0,0,0,0,0]
    elif word_label =='Hugo_Chavez':
        return [0,0,0,0,1,0,0,0,0,0,0]
    elif word_label =='Jacques_Chirac':
        return [0,0,0,0,0,1,0,0,0,0,0]
    elif word_label =='Jean_Chretien':
        return [0,0,0,0,0,0,1,0,0,0,0]
    elif word_label =='John_Aschcroft':
        return [0,0,0,0,0,0,0,1,0,0,0]
    elif word_label =='Junichiro_Koizumi':
        return [0,0,0,0,0,0,0,0,1,0,0]
    elif word_label =='Serena_Williams':
        return [0,0,0,0,0,0,0,0,0,1,0]
    elif word_label =='Tony_Blair':
        return [0,0,0,0,0,0,0,0,0,0,1]

#%%
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        if label==None:
            continue
        path = os.path.join(Train_dir,img)
        #print(path,label)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(Img_size,Img_size))
        #(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        training_data.append([np.array(img),np.array(label)])
        #training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #np.save('sc_train_data.npy',training_data) 
    return training_data
X=create_training_data()
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

history = autoencoder.fit(x_train,x_train,epochs=1,batch_size=16,shuffle=True,
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
    decoded=decoded_imgs[i].reshape((64,64))
    #cv2.imshow('output',decoded)
    #cv2.waitKey(0)
    #cv2.imwrite('AE_XRay/xray.'+ str(i)+'.jpeg',255*decoded)
    #fig.savefig('face'+ str(i)+ '.jpg')
    plt.show()
    
#%%
# Function to use different shapes in plot using list
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
# this block assigns color to each class in the plot
test1 = []
test_shape = y_test.shape
def convert_to_1d():
    
    #testvalue=[]
    for i in range(test_shape[0]):
        if (y_test[i] == np.array([1,0,0,0,0,0,0,0,0,0,0])).all():
            test1.append(0)
        elif (y_test[i] == np.array([0,1,0,0,0,0,0,0,0,0,0])).all():
            test1.append(1)
        elif (y_test[i] == np.array([0,0,1,0,0,0,0,0,0,0,0])).all():
            test1.append(2)
        elif (y_test[i] == np.array([0,0,0,1,0,0,0,0,0,0,0])).all():
            test1.append(3)
        elif (y_test[i] == np.array([0,0,0,0,1,0,0,0,0,0,0])).all():
            test1.append(4)
        elif (y_test[i] == np.array([0,0,0,0,0,1,0,0,0,0,0])).all():
            test1.append(5)
        elif (y_test[i] == np.array([0,0,0,0,0,0,1,0,0,0,0])).all():
            test1.append(6)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,1,0,0,0])).all():
            test1.append(7)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,1,0,0])).all():
            test1.append(8)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,0,1,0])).all():
            test1.append(9)
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,0,0,1])).all():
            test1.append(10)
            
    return test1
        
test2 = convert_to_1d()
#%%
#creating marker list for the plot
markershape = []
test_shape = y_test.shape
markers = ['.','o','v','^','>','<','s','p','*','h','H']
def convert_to_markershape():
    
    #testvalue=[]
    for i in range(test_shape[0]):
        if (y_test[i] == np.array([1,0,0,0,0,0,0,0,0,0,0])).all():
            markershape.append(markers[0])
        elif (y_test[i] == np.array([0,1,0,0,0,0,0,0,0,0,0])).all():
            markershape.append(markers[1])
        elif (y_test[i] == np.array([0,0,1,0,0,0,0,0,0,0,0])).all():
            markershape.append(markers[2])
        elif (y_test[i] == np.array([0,0,0,1,0,0,0,0,0,0,0])).all():
            markershape.append(markers[3])
        elif (y_test[i] == np.array([0,0,0,0,1,0,0,0,0,0,0])).all():
            markershape.append(markers[4])
        elif (y_test[i] == np.array([0,0,0,0,0,1,0,0,0,0,0])).all():
            markershape.append(markers[5])
        elif (y_test[i] == np.array([0,0,0,0,0,0,1,0,0,0,0])).all():
            markershape.append(markers[6])
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,1,0,0,0])).all():
            markershape.append(markers[7])
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,1,0,0])).all():
            markershape.append(markers[8])
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,0,1,0])).all():
            markershape.append(markers[9])
        elif (y_test[i] == np.array([0,0,0,0,0,0,0,0,0,0,1])).all():
            markershape.append(markers[10])
            
    return markershape
        
markershape1 = convert_to_markershape()
#%%
#Encoder dimension (550, 3600)
#decoder dimesnion (550, 4096)
plt.subplots()
mscatter(encoded_imgs[:,0], encoded_imgs[:,1], c = test2, m=markershape1, s=100, edgecolors='black')
#%%
ys = np.unique(test2)
means = np.array([np.mean(encoded_imgs[test2 == y, :], axis=0) for y in ys])
plt.scatter(means[:, 0], means[:, 1], c=ys, s=200, edgecolors='black')
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()