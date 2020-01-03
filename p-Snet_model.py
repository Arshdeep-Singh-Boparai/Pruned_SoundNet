#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 16:01:37 2020

@author: arshdeep
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D,AveragePooling1D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
#import theano 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
import tensorflow as tf
from keras.models import Model
from keras.models import load_model

#%%
weight_path='/home/arshdeep/Signal_procesing letters/p-snet_Model/p-Snet_weights.hdf5'
#%%

model =Sequential()

model.add(Conv1D(16,64,strides=2,input_shape=(None,1))) #layer1
model.add(ZeroPadding1D(padding=16))
model.add(BatchNormalization()) #layer2
convout1= Activation('relu')
model.add(convout1) #layer3


#initia_weights1=[np.reshape(param_G['conv1']['weights'],(64,1,16)),param_G['conv1']['biases'],param_G['conv1']['gamma'],param_G['conv1']['beta'],param_G['conv1']['mean'],param_G['conv1']['var'],np.reshape(param_G['conv2']['weights'],(32,16,32)),param_G['conv2']['biases'],param_G['conv2']['gamma'],param_G['conv2']['beta'],param_G['conv2']['mean'],param_G['conv2']['var'],np.reshape(param_G['conv3']['weights'],(16,32,64)),param_G['conv3']['biases'],param_G['conv3']['gamma'],param_G['conv3']['beta'],param_G['conv3']['mean'],param_G['conv3']['var'],np.reshape(param_G['conv4']['weights'],(8,64,128)),param_G['conv4']['biases'],param_G['conv4']['gamma'],param_G['conv4']['beta'],param_G['conv4']['mean'],param_G['conv4']['var'],np.reshape(param_G['conv5']['weights'],(4,128,256)),param_G['conv5']['biases'],param_G['conv5']['gamma'],param_G['conv5']['beta'],param_G['conv5']['mean'],param_G['conv5']['var'],np.reshape(param_G['conv6']['weights'],(4,256,512)),param_G['conv6']['biases'],param_G['conv6']['gamma'],param_G['conv6']['beta'],param_G['conv6']['mean'],param_G['conv6']['var'],np.reshape(param_G['conv7']['weights'],(4,512,1024)),param_G['conv7']['biases'],param_G['conv7']['gamma'],param_G['conv7']['beta'],param_G['conv7']['mean'],param_G['conv7']['var'],np.reshape(param_G['conv8']['weights'],(8,1024,1000)),param_G['conv8']['biases'],np.reshape(param_G['conv8_2']['weights'],(8,1024,401)),param_G['conv8_2']['biases']]

model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer4
#
#
model.add(Conv1D(32,32,strides=2)) #layer5
model.add(ZeroPadding1D(padding=8))
model.add(BatchNormalization()) #layer6
convout2= Activation('relu')
model.add(convout2) #layer7
#model.add(Dropout(0.5))

model.add(MaxPooling1D(pool_size=8, padding='valid')) #layer8

model.add(Conv1D(41,16,strides=2)) #layer9
model.add(ZeroPadding1D(padding=4))
model.add(BatchNormalization()) #layer10
convout3= Activation('relu')
model.add(convout3) #layer11
#model.add(Dropout(0.5))

model.add(Conv1D(56,8,strides=2)) #layer12
model.add(ZeroPadding1D(padding=2))
model.add(BatchNormalization()) #layer13
convout4= Activation('relu')
model.add(convout4) #layer14


model.add(Conv1D(88,4,strides=2)) #layer15
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer16
convout5= Activation('relu')
model.add(convout5) #layer17

model.add(MaxPooling1D(pool_size=4,padding='valid')) #layer18

model.add(Conv1D(235,4,strides=2)) #layer15
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer16
convout6= Activation('relu')
model.add(convout6) #layer17
#model.add(Dropout(0.5))
#model.set_weights(initia_weights1)
model.add(Conv1D(59,4,strides=2)) #layer18
model.add(ZeroPadding1D(padding=1))
model.add(BatchNormalization()) #layer19
convout7= Activation('relu')
model.add(convout7) #layer2/home/surbhi/surbhi/DATA_BABY0
#model.add(Dropout(0.5))

model.load_weights(weight_path)