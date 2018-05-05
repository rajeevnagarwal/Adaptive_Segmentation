import tensorflow as tf
import numpy as np
from math import *

def get_weights(name,shape):
	#print(name)
	return tf.get_variable(name+"_wht",shape,initializer = tf.truncated_normal_initializer(stddev=0.05),dtype=tf.float32)

def get_biases(name,size):
	return tf.get_variable(name+"_bs",size,initializer=tf.constant_initializer(0.05),dtype=tf.float32)
	
def relu_activation(layer):
	#return leaky_relu_activation(layer)
	return tf.nn.relu(layer)

def sigmoid_activation(layer):
	return tf.nn.sigmoid(layer)

def max_pool_layer(layer):
	return tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def batch_normalization(input):
	#print(input)
	layer = tf.layers.batch_normalization(input,momentum = 0.99,epsilon=0.001,center=True,scale = True)
	return layer
def conv_layer(input,stride,num_channels,conv_filter_size,n_filters,name):

	weights = get_weights(name,shape=[conv_filter_size , conv_filter_size , num_channels , n_filters] )
	biases = get_biases(name,n_filters)


	layer = tf.nn.conv2d(input = input , filter = weights, strides = [1,stride,stride,1] , padding='SAME')
	layer = layer + biases


	return layer
	
def gaussian_noise(input,std):
	noise = tf.random_normal(shape=tf.shape(input),mean=0.0,stddev=std,dtype=tf.float32)
	return input+noise

def leaky_relu_activation(input,slope=0.2):
	#ans = tf.nn.relu(input)- slope*tf.nn.relu(-input) 
	ans = tf.nn.leaky_relu(input,alpha=slope)
	return ans

def fc_layer(input,num_inputs,num_outputs,name):
	weight = get_weights(name,shape=[num_inputs,num_outputs])
	bias = get_biases(name,num_outputs)
	ans = tf.matmul(input,weight)+bias
	return ans

def flatten_layer(input):
	shape = input.get_shape()
	n_features = shape[1:4].num_elements()
	layer = tf.reshape(input,[-1,n_features])
	return layer

def dropout_layer(input,name,keep_prob=0.9):
	return tf.nn.dropout(input,keep_prob = keep_prob,name=name)

'''def deconv_layer(input,filters,k_size,strides,name):
	k_size = (k_size);
	k_size = (k_size , k_size)

	strides = (strides);
	strides = (strides, strides)
	ans = tf.layers.conv2d_transpose(input,filters = int(filters),kernel_size = k_size,strides = strides,padding='same',name=name)
	return ans'''

def deconv_layer(input,in_channel,out_channel,k_size,out_shape,strides,name,padding):
	filter_shape = [k_size[0],k_size[1],out_channel,in_channel]
	#print(name)
	w = get_deconv_filter(filter_shape,name)
	s = [1,strides,strides,1]
	ans = tf.nn.conv2d_transpose(value=input,filter=w,output_shape=out_shape,strides = s,padding=padding,name=name)
	return ans

def get_deconv_filter(f_shape,name):
	width = f_shape[0]
	height = f_shape[1]
	f = ceil(width/2.0)
	c = (2*f-1-f%2)/(2.0*f)
	bilinear = np.zeros([f_shape[0],f_shape[1]])
	for x in range(width):
		for y in range(height):
			value = (1-abs(x/f-c))*(1-abs(y/f-c))
			bilinear[x,y]= value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:,:,i,i] = bilinear
	init = tf.constant_initializer(value=weights,dtype=tf.float32)
	return tf.get_variable(name=name+"_up_filter",initializer = init,shape=weights.shape)
