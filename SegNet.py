import tensorflow as tf
from nn_utils import *
import numpy as np

def SegNet(images,batch_size,num_classes,name):
	shape = images.get_shape()
	with tf.variable_scope("SegNet_"+name):
		with tf.variable_scope("encoder_"+name):
			conv1 = conv_layer(images,stride = 1,num_channels = shape[3],conv_filter_size=7,n_filters=64,name="conv1")
			conv1 = batch_normalization(conv1)
			conv1 = max_pool_layer(conv1)
			#print(conv1)
			conv2 = conv_layer(conv1,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv2")
			conv2 = batch_normalization(conv2)
			conv2 = max_pool_layer(conv2)
			#print(conv2)
			conv3 = conv_layer(conv2,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv3")
			conv3 = batch_normalization(conv3)
			conv3 = max_pool_layer(conv3)
			#print(conv3)
			#conv4 = conv_layer(conv3,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv4")
			#conv4 = batch_normalization(conv4)
			#conv4 = max_pool_layer(conv4)
			#print(conv4)
			#conv5 = conv_layer(conv4,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv5")
			#conv5 = batch_normalization(conv5)
			#conv5 = max_pool_layer(conv5)
			#print(conv5)

		with tf.variable_scope("decoder_"+name):
			upsample5 = deconv_layer(input=conv3,in_channel=64,out_channel=64,k_size=(2,2),out_shape=[batch_size,45,60,64],strides=2,name="up5",padding="SAME")
			conv_decode5 = conv_layer(upsample5,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode5")
			conv_decode5 = batch_normalization(conv_decode5)
			#conv_decode5 = max_pool_layer(conv_decode5)
			#print(upsample5)
			#print(conv_decode5)
			upsample4 = deconv_layer(input = conv_decode5,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,90,120,64],strides=2,name="up4",padding="SAME")
			conv_decode4 = conv_layer(upsample4,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode4")
			conv_decode4 = batch_normalization(conv_decode4)
			#conv_decode4 = max_pool_layer(conv_decode4)
			#print(upsample4)
			#print(conv_decode4)
			upsample3 = deconv_layer(input =conv_decode4,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,180,240,64],strides=2,name="up3",padding="SAME")
			conv_decode3 = conv_layer(upsample3,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode3")
			conv_decode3 = batch_normalization(conv_decode3)
			#conv_decode3 = max_pool_layer(conv_decode3)
			#print(upsample3)
			#print(conv_decode3)
			#upsample2 = deconv_layer(input=conv_decode3,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,360,480,64],strides=2,name="up2",padding="SAME")
			#conv_decode2 = conv_layer(upsample2,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode2")
			#conv_decode2 = batch_normalization(conv_decode2)
			#conv_decode2 = max_pool_layer(conv_decode2)
			#print(upsample2)
			#print(conv_decode2)
			#upsample1 = deconv_layer(input = conv_decode2,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,720,960,64],strides=2,name="up1",padding="SAME")
			#conv_decode1 = conv_layer(upsample1,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode1")
			#conv_decode1 = batch_normalization(conv_decode1)
			#conv_decode1 = max_pool_layer(conv_decode1)
			#print(upsample1)
			#print(conv_decode1)
			#upsample0 = deconv_layer(input = conv_decode1,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,1440,1920,64],strides=2,name="up0",padding="SAME")
			#conv_decode0 = conv_layer(upsample0,stride=1,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode0")
			#conv_decode0 = batch_normalization(conv_decode0)
			#conv_decode0 = max_pool_layer(conv_decode0)
			#print(upsample0)
			#print(conv_decode0)
			#upsample_1 = deconv_layer(input = conv_decode0,in_channel=64,out_channel=64,k_size = (2,2),out_shape=[batch_size,2880,3840,64],strides=2,name="up_1",padding="SAME")
			#conv_decode_1 = conv_layer(upsample_1,stride=2,num_channels=64,conv_filter_size=7,n_filters=64,name="conv_decode_1")
			#conv_decode_1 = batch_normalization(conv_decode_1)
			#conv_decode_1 = max_pool_layer(conv_decode_1)
			#print(upsample_1)
			#print(conv_decode_1)
			conv_final = conv_layer(conv_decode3,stride=1,num_channels=64,conv_filter_size=1,n_filters=num_classes,name="conv_final")
			#print(conv_final)
			conv_final = tf.nn.softmax(conv_final)
			return conv_final			
		pass
if __name__=="__main__":
	pass