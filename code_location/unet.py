from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Lambda,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

ss=10

def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.greater_equal(y_true_f,-0.5)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def mse_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
#    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    loss=K.mean(K.square(y_true_f - y_pred_f))
    return loss

def abse(y_true, y_pred):
   return K.mean(K.abs(y_pred - y_true))

def huber_loss(y_true, y_pred, clip_delta=10):
  error = y_true - y_pred
  cond  = K.abs(error) < clip_delta

  squared_loss = 0.5 * K.square(error)
  linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return K.mean(huber_loss(y_true, y_pred, clip_delta))


def pcc(layer_in, num_filter, size_kernel, activation='relu', padding='same'):
    x = MaxPooling2D(pool_size=2)(layer_in)
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def ucc(layer_in1,layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([Conv2DTranspose(num_filter,2,strides=2,padding=padding)(layer_in1), layer_in2], axis=3)
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def get_unet(the_lr=1e-3, num_class=15, num_channel=1, size=(1024,1024)):
#    inputs = Input((None, None, num_channel))
    inputs = Input((size[0], size[1], num_channel)) 
    print(inputs.shape)

    num_blocks=4 # 128*25 -> 200
    initial_filter=num_channel * 3 #15
    scale_filter=2
    size_kernel=7
    activation='relu'
    padding='same'    

    layer_down=[]
    layer_up=[]

    conv0 = BatchNormalization()(Conv2D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(inputs))
    conv0 = BatchNormalization()(Conv2D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(conv0))

    layer_down.append(conv0)
    num=initial_filter
    for i in range(num_blocks):
        num=int(num * scale_filter)
        the_layer=pcc(layer_down[i], num, size_kernel, activation=activation, padding=padding)
        layer_down.append(the_layer)

    layer_up.append(the_layer)
    for i in range(num_blocks):
        num=int(num / scale_filter)
        the_layer=ucc(layer_up[i],layer_down[-(i+2)],num, size_kernel, activation=activation, padding=padding)
        layer_up.append(the_layer)

    #convn = Conv2D(num_class, 1, activation='relu')(layer_up[-1]) 
    convn = Conv2D(num_class, 1, activation='sigmoid')(layer_up[-1]) 

    model = Model(inputs=[inputs], outputs=[convn])

    model.compile(optimizer=Adam(lr=the_lr,beta_1=0.9, beta_2=0.999,decay=1e-5), loss=crossentropy_cut, metrics=[abse])

    return model


