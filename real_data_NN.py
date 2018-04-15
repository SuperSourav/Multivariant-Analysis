from collections import namedtuple
from scipy.stats import norm
import dynamic_learning_rate
import tensorflow as tf
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

with open ('./input data/all_test_x', 'rb') as fp:
    test_x = pickle.load(fp)

with open('./input data/max_min_features', 'rb') as fp:
	temp = pickle.load(fp)
	min_features = temp[0]
	max_features = temp[1]

if __name__ == '__main__':
	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	test = np.array(test_x).astype(np.float32)
	# test = test.reshape(1,21)
	print(test.shape)

	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('./output_data/3-layer_all_data_SGDR_lr.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./output_data/'))
		temp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for item in temp:
			print(item)
		w1 = sess.run('Variable_1:0')
		b1 = sess.run('Variable_2:0')

		w2 = sess.run('Variable_3:0')
		b2 = sess.run('Variable_4:0')
		scale2 = sess.run('Variable_5:0')
		beta2 = sess.run('Variable_6:0')
		
		w3 = sess.run('Variable_7:0')
		b3 = sess.run('Variable_8:0')
		scale3 = sess.run('Variable_9:0')
		beta3 = sess.run('Variable_10:0')

		w_out = sess.run('Variable_11:0')
		b_out = sess.run('Variable_12:0')

		out = tf.nn.sigmoid(tf.add(tf.matmul(test,w1),b1))
		
		out = tf.matmul(out,w2)
		mean, var = tf.nn.moments(out,[0])
		out = tf.nn.sigmoid(tf.nn.batch_normalization(out,mean,var,beta2,scale2,1e-3))
		
		out = tf.matmul(out,w3)
		mean, var = tf.nn.moments(out,[0])
		out = tf.nn.sigmoid(tf.nn.batch_normalization(out,mean,var,beta3,scale3,1e-3))
		
		out = tf.nn.softmax(tf.add(tf.matmul(out,w_out),b_out))

		print("NN structure: ", w1.shape, w2.shape, w3.shape)

		signal_masses = []
		background_masses = []

		# print(sess.run(tf.nn.softmax(out)))
		# print(tf.nn.softmax(out))

		signal_probability = sess.run(out[:,0])
		print("signal probability = ", signal_probability)

		for i in range(int(len(test))):
			background_masses.append(test_x[i][3]*(max_features[3]-min_features[3])+min_features[3]/1000)
			if signal_probability[i] > 0.35:
				signal_masses.append(test_x[i][3]*(max_features[3]-min_features[3])+min_features[3]/1000)

		print("Filtered = ", len(signal_masses), "events")

		plt.figure(1)
		plt.hist(signal_masses, 100, histtype='step', label = "Filtered", stacked = True, density = 1)
		plt.hist(background_masses, 100, histtype='step', label = "Unfiltered", stacked = True, density = 1)
		plt.xlabel("Mass of Highest Pt Jet [GeV]")
		plt.legend(loc = "upper right")
		plt.title("Filtered Jet Mass all data, Threshold = 0.35, SGDR_lr")
		plt.show()
		plt.close(1)









