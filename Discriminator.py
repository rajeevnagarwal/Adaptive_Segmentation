import tensorflow as tf
import numpy as np
from nn_utils import *

def Discriminator(images,name):
	shape = images[0].get_shape()
	#print(shape)
	noise = 0.9
	with tf.variable_scope("Discriminator_"+name):
		conv1 = conv_layer(images,stride=1,num_channels=shape[2],conv_filter_size=3,n_filters=64,name="conv1")
		conv1 = dropout_layer(conv1,keep_prob=noise,name="drop1")
		conv1 = batch_normalization(conv1)
		conv1 = leaky_relu_activation(conv1)
		#conv1 = gaussian_noise(conv1,std=0.2)
		
		conv2 = conv_layer(conv1,stride=2,num_channels=64,conv_filter_size=3,n_filters=128,name="conv2")
		conv2 = dropout_layer(conv2,keep_prob=noise,name="drop2")
		conv2 = batch_normalization(conv2)
		conv2 = leaky_relu_activation(conv2)
		#conv2 = gaussian_noise(conv2,std=0.2)
		
		conv3 = conv_layer(conv2,stride=2,num_channels=128,conv_filter_size=3,n_filters=256,name="conv3")
		conv3 = dropout_layer(conv3,keep_prob=noise,name="drop3")
		conv3 = batch_normalization(conv3)
		conv3 = leaky_relu_activation(conv3)
		#conv3 = gaussian_noise(conv3,std=0.2)
		
		conv4 = conv_layer(conv3,stride=2,num_channels=256,conv_filter_size=3,n_filters=512,name="conv4")
		conv4 = dropout_layer(conv4,keep_prob=noise,name="drop4")
		conv4 = batch_normalization(conv4)
		conv4 = leaky_relu_activation(conv4)
		#conv4 = gaussian_noise(conv4,std=0.2)

		#conv5 = conv_layer(conv4,stride=2,num_channels=512,conv_filter_size=3,n_filters=1024,name="conv5")
		#conv5 = dropout_layer(conv5,keep_prob=0.9,name="drop5")
		#conv5 = batch_normalization(conv5)
		#conv5 = leaky_relu_activation(conv5)
		#conv5 = gaussian_noise(conv5,std=0.2)
		
		conv_flat = flatten_layer(conv4)
		shape_flat = conv_flat.get_shape()
		fc1 = fc_layer(conv_flat,shape_flat[1],1,name="fc1")
		ans = ((fc1))
		return ans
		pass

if __name__=="__main__":
	pass
