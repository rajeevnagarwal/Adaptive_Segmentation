import tensorflow as tf
import numpy as np
import os
import cv2
from Generator import *
from Discriminator import *
from SegNet import *
from nn_utils import *

from global_gen import *
#from pixel_domain_adaptation.utils.indian_road_gen import *
#from pixel_domain_adaptation.utils.synthia_gen import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def train(learning_rate,batch_size,n_iter,image_width,image_height,num_classes,z_size):
	x_s = tf.placeholder(shape=[batch_size,image_width,image_height,3],name='x_s',dtype=tf.float32)
	x_t = tf.placeholder(name='x_t',shape=[batch_size,image_width,image_height,3],dtype=tf.float32)
	y_s = tf.placeholder(name='y_s',shape=[batch_size,image_width,image_height,num_classes],dtype=tf.float32)
	z = tf.placeholder(shape=[1,z_size],dtype=tf.float32,name='z')
	dis_tar = Discriminator(x_t,name="dis_tar")
	dis_gen = Discriminator(Generator(x_s,z,name="gen1"),name="dis_gen")
	gen2 = Generator(x_s,z,name="gen2")
	#print(gen2)
	cla_gen = SegNet(gen2,batch_size,num_classes,name="cla_gen")
	cla_source = SegNet(x_s,batch_size,num_classes,name="cla_source")
	epsilon = np.finfo(np.float32).eps
	#if(tf.assert_less_equal(dis_tar,epsilon)dis_tar<=epsilon):
	#	dis_tar = dis_tar + epsilon
	#if(dis_gen-1<=epsilon):
	#	dis_gen= dis_gen - epsilon

	loss_domain = tf.reduce_mean(tf.log_sigmoid((dis_tar)))+tf.reduce_mean(tf.log_sigmoid((1.0-dis_gen)))
	#print(cla_gen)
	#print(cla_source)
	loss_classifier = tf.reduce_mean(tf.reduce_sum(-y_s*tf.log(cla_gen),axis=[1]))-tf.reduce_mean(tf.reduce_sum(y_s*tf.log(cla_source),axis=[1]))
	t_vars = tf.trainable_variables()
	class_vars = [var for var in t_vars if 'SegNet' in var.name]
	dis_vars = [var for var in t_vars if 'Discriminator' in var.name]
	gen_vars = [var for var in t_vars if 'Generator' in var.name]
	d_t = class_vars + dis_vars
	domain_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss_domain,var_list = d_t)
	class_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss_classifier,var_list = gen_vars)
	model_path = './model/'
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()
	interval = 20

	writer = tf.summary.FileWriter(logdir='logdir', graph=tf.get_default_graph())
	writer.flush()

	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		flag = True
		for i in range(n_iter):
			#gbl = GBL(batch_size)
			if flag==True:
				gbl = GBL(batch_size)
				avg_l1 = 0
				for iCombined in gbl.global_db_gen():
					#iCombined = np.array(iCombined)
					z_b = tf.random_uniform(minval=-1,maxval=1,shape=[1,z_size])
					z_b = sess.run(z_b)
					s_i = np.array([x for x,y,z in iCombined])
					t_i = np.array([y for x,y,z in iCombined])
					s_l = np.array([z for x,y,z in iCombined])
					#s_i = cv2.resize(s_i,(image_width,image_height))

					#s_l = cv2.resize(s_l, (image_width, image_height))
					#t_i = cv2.resize(t_i, (image_width, image_height))

					#break
					#print(iCombined)
					s_i = s_i/255.0
					s_i = s_i -np.mean(s_i)
					s_i = s_i/(np.std(s_i,axis=0) + np.finfo(np.float32).eps)
					#s_i = s_i/255.0
					t_i = t_i/255.0
					t_i = t_i - np.mean(t_i)
					t_i = t_i/(np.std(t_i,axis=0)  + np.finfo(np.float32).eps)
					#t_i = s_i/255.0
				#print(s_i.shape)
				#print(t_i.shape)
					d_t,d_g,l1,_ = sess.run([dis_tar,dis_gen,loss_domain,domain_optimizer],feed_dict={x_s:s_i,x_t:t_i,y_s:s_l,z:z_b})
					#if ( i % 5 == s0 ):
					#print(l1)
					avg_l1 = avg_l1 + l1
					avg_l1 = avg_l1/60.0
					#flag = 1
					print("Epoch: "+str(i)+" Domain Loss: "+str(l1))

			if flag==True:
				gbl = GBL(batch_size)
				avg_l2 = 0
				for iCombined in gbl.global_db_gen():
					# iCombined = np.array(iCombined)
					z_b = tf.random_normal(shape=[1, z_size])
					z_b = sess.run(z_b)
					s_i = np.array([x for x, y, z in iCombined])
					t_i = np.array([y for x, y, z in iCombined])
					s_l = np.array([z for x, y, z in iCombined])
					#if (np.random.random_sample() < 0.5):
					#	print("breaking")
					#	break
					s_i = s_i / 255.0
					s_i = s_i - np.mean(s_i)
					s_i = s_i / (np.std(s_i, axis=0) + np.finfo(np.float32).eps)
					# s_i = s_i/255.0
					t_i = t_i / 255.0
					t_i = t_i - np.mean(t_i)
					t_i = t_i / (np.std(t_i, axis=0) + np.finfo(np.float32).eps)
				# print(s_i.shape)
				# print(t_i.shape)
					l2,_ = sess.run([loss_classifier, class_optimizer], feed_dict={x_s: s_i, x_t: t_i, y_s: s_l, z: z_b})
					avg_l2 = avg_l2 + l2
					#flag = 0
					avg_l2 = avg_l2/60.0
					print("Epoch: "+str(i)+" Classifier Loss: "+str(l2))
			if((i+1)%interval==0):
				saver.save(sess,os.path.join(model_path,'model'),global_step=i+1)
		saver.save(sess,os.path.join(model_path,'model'),global_step=i+1)
if __name__=="__main__":
	learning_rate = 0.0001
	batch_size = 20
	n_iter = 2000
	image_width = 180
	image_height = 240
	num_classes = 12
	z_size = 10
	train(learning_rate,batch_size,n_iter,image_width,image_height,num_classes,z_size)
	pass
