import tensorflow as tf
import numpy as np
import os
import cv2
from Generator import *
from Discriminator import *
from SegNet import *
from nn_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = "test_images/"
#path = '/home/himanshu/DL/Project/Dataset/'
model_path = "model/"
model_name = "model-160"
model_name_meta = "model-160.meta"
color_dict = {0:(0,0,0),1:(128,128,128),2:(128,0,0),3:(128,64,128),4:(0,0,192),5:(64,64,128),6:(128,128,0),7:(192,192,128),8:(64,0,128),9:(192,128,128),10:(64,64,0),11:(0,128,192)}


def load_images(path, image_width, image_height):
	filenames = sorted(os.listdir(path))
	X = []
	for file in filenames:
		if 'png' in file:
			file = path + "/" + file;
			img = cv2.imread(file)
			img = cv2.resize(img, (image_height, image_width))
			X.append(img)
	X = np.array(X)


	return X


def test(batch_size, image_width, image_height, num_classes, z_size):
    # load_images(path,  image_width , image_height)[0: 20]
	learning_rate = 0.00001
	x_s = tf.placeholder(shape=[batch_size, image_width, image_height, 3], name='x_s', dtype=tf.float32)

	x_t = tf.placeholder(name='x_t', shape=[batch_size, image_width, image_height, 3], dtype=tf.float32)
	y_s = tf.placeholder(name='y_s', shape=[batch_size, image_width, image_height, num_classes], dtype=tf.float32)
	z = tf.placeholder(shape=[1, z_size], dtype=tf.float32, name='z')
	dis_tar = Discriminator(x_t, name="dis_tar")
	dis_gen = Discriminator(Generator(x_s, z, name="gen1"), name="dis_gen")
	gen2 = Generator(x_s, z, name="gen2")
	# print(gen2)
	cla_gen = SegNet(gen2, batch_size, num_classes, name="cla_gen")
	cla_source = SegNet(x_s, batch_size, num_classes, name="cla_source")
	epsilon = np.finfo(np.float32).eps
	loss_domain = tf.reduce_mean(tf.log_sigmoid((dis_tar))) + tf.reduce_mean(tf.log_sigmoid((1.0 - dis_gen)))
	# print(cla_gen)
	# print(cla_source)
	loss_classifier = tf.reduce_mean(tf.reduce_sum(-y_s * tf.log(cla_gen) - y_s * tf.log(cla_source), axis=[1]))
	t_vars = tf.trainable_variables()
	class_vars = [var for var in t_vars if 'SegNet' in var.name]
	dis_vars = [var for var in t_vars if 'Discriminator' in var.name]
	gen_vars = [var for var in t_vars if 'Generator' in var.name]
	d_t = class_vars + dis_vars
	domain_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_domain,var_list = d_t)
	class_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_classifier,var_list = gen_vars)

	saver = tf.train.Saver();
	coord = tf.train.Coordinator();

	print("Aa gya")

	with tf.Session(config=config) as sess:
		new_saver = tf.train.import_meta_graph(os.path.join(model_path, model_name_meta))
		new_saver.restore(sess, os.path.join(model_path, model_name))
		list_of_op = tf.get_default_graph().get_operations()
		ops_2 = tf.get_default_graph().get_operation_by_name('SegNet_cla_source/decoder_cla_source/Softmax');
		#		ops_1 = tf.get_default_graph().get_operation_by_name('SegNet_cla_source/decoder_cla_source/Reshape_1');
		#x_s = tf.get_default_graph().get_tensor_by_name('x_s:0')
		# tf.get_default_graph().get_tensor_by_name()

		# collection = tf.get_collection('SegNet_cla_source/decoder_cla_source')
		#

		s_i = load_images(path, image_width, image_height)[50: 70]
		for k in range(len(s_i)):
			cv2.imwrite("s_i" + str(k) + ".png", s_i[k])
		s_i = s_i / 255.0
		s_i = s_i - np.mean(s_i)
		s_i = s_i / (np.std(s_i, axis=0) + np.finfo(np.float32).eps)

		print("Aa gya 2", np.shape(s_i))

		#
		# for index in range(np.shape(X)[0]):
		# 	pass
		batch_output = sess.run(cla_source, feed_dict={x_s: s_i})
		#s_i = s_i*(np.std(s_i,axis=0)+np.finfo(np.float32).eps)
		#s_i = s_i+np.mean(s_i)
		#s_i = s_i*255
		for k in range(len(batch_output)):
			op_imag = batch_output[k]
			s_i = s_i*255
			print(s_i.shape)
			op_imag = op_imag;
			#print(op_imag[150][10])
			op_imag = getClass(op_imag)
			#print(op_imag)
			cv2.imwrite("op_image"+str(k)+".png",op_imag);
			#cv2.imwrite("s_i"+str(k)+".png",s_i[k])
def getClass(image):
	s = image.shape
	new_image = np.zeros(shape=(s[0],s[1],3))
	sum = 0
	for i in range(s[0]):
		for j in range(s[1]):
			index = np.argmax(image[i][j])
			if(index>=0):
				sum = sum+index
				new_image[i][j][0] = color_dict[index][0]
				new_image[i][j][1] = color_dict[index][1]
				new_image[i][j][2] = color_dict[index][2]
			else:
				index = 0
				sum = sum+index
				new_image[i][j][0] = color_dict[index][0]
				new_image[i][j][1] = color_dict[index][1]
				new_image[i][j][2] = color_dict[index][2]

	print(sum)
	return new_image
if __name__ == "__main__":
    image_width = 180
    image_height = 240
    num_classes = 12
    z_size = 10
    batch_size = 20
    test(batch_size, image_width, image_height, num_classes, z_size)
