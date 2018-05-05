import tensorflow as tf
import numpy as np
from nn_utils import *

def Generator(images,noise,name):
	shape = images.get_shape()
	with tf.variable_scope("Generator_"+name):
		#print(shape)
		#print(noise.shape)
		num_out = shape[0]*shape[1]*shape[2]
		#print(num_out)
		fc1 = fc_layer(noise,noise.shape[1],num_out,name="fc1_gen")
		fc1 = tf.reshape(fc1,[shape[0],shape[1],shape[2],1])
		images = tf.concat([images,fc1],axis=3)
		shape = images.get_shape()
		
		
		conv1 = conv_layer(images,stride=1,num_channels=shape[3],conv_filter_size=3,n_filters=64,name="conv1_gen")
		conv1 = relu_activation(conv1)
		#Residual Block
		conv2 = conv_layer(conv1,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv2_gen")
		conv2 = batch_normalization(conv2)
		conv2 = relu_activation(conv2)
		conv3 = conv_layer(conv2,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv3_gen")
		conv3 = batch_normalization(conv3)
		conv3 = conv3 + conv1
		#Residual Block
		conv4 = conv_layer(conv3,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv4_gen")
		conv4 = batch_normalization(conv4)
		conv4 = relu_activation(conv4)
		conv5 = conv_layer(conv4,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv5_gen")
		conv5 = batch_normalization(conv5)
		conv5 = conv5 + conv3
		#Residual Block
		conv6 = conv_layer(conv5,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv6_gen")
		conv6 = batch_normalization(conv6)
		conv6 = relu_activation(conv6)
		conv7 = conv_layer(conv6,stride=1,num_channels=64,conv_filter_size=3,n_filters=64,name="conv7_gen")
		conv7 = batch_normalization(conv7)
		#print(conv7)
		conv7 = conv7 + conv5
		#print(conv7)		
		conv8 = conv_layer(conv7,stride=1,num_channels=64,conv_filter_size=3,n_filters=3,name="conv8_gen")
		
		res = tf.tanh(conv8);
		
		return res;
		
		
