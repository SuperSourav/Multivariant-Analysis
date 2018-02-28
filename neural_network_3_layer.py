from collections import namedtuple
import tensorflow as tf
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

with open ('./pickled data/%s_train_x' % sys.argv[1], 'rb') as fp:
    train_x = pickle.load(fp)

with open ('./pickled data/%s_train_y' % sys.argv[1], 'rb') as fp:
    train_y = pickle.load(fp)

with open ('./pickled data/%s_test_x' % sys.argv[1], 'rb') as fp:
    test_x = pickle.load(fp)

with open ('./pickled data/%s_test_y' % sys.argv[1], 'rb') as fp:
    test_y = pickle.load(fp)

n_classes = 2
batch_size = 1000
max_epochs = int(sys.argv[3])
divisions = 20

x = tf.placeholder('float',[None, len(train_x[0])])	
y = tf.placeholder('float',[None, len(train_y[0])])
MyStruct = namedtuple("MyStruct", "roc auc threshold_plot filtered_mass name")

def neural_network_model(data, layer_sizes):

	hidden_1_layer = {'f_fum':layer_sizes[0],
				  'weight':tf.Variable(tf.random_normal([len(train_x[0]), layer_sizes[0]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[0]]))}

	hidden_2_layer = {'f_fum':layer_sizes[1],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[0], layer_sizes[1]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[1]]))}

	hidden_3_layer = {'f_fum':layer_sizes[2],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[1], layer_sizes[2]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[2]]))}

	output_layer = {'f_fum':None,
				'weight':tf.Variable(tf.random_normal([layer_sizes[2], n_classes])),
				'bias':tf.Variable(tf.random_normal([n_classes])),}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.sigmoid(l3)

	output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

	#output = tf.nn.softmax(output)

	return output

# Generate the confusion matrix for a binary classifier
def confusion_matrix(prediction, sample_x, sample_y, threshold):
	true_signal = 0
	false_signal = 0
	false_background = 0
	true_background = 0
	filtered_mass = []
	# print("Threshold = ", threshold)
	i = 0
	while i < len(sample_y):
		signal_probability = prediction[i][0]
		if signal_probability>threshold and sample_y[i][0]==1 :
			true_signal += 1
			filtered_mass.append(sample_x[i][3])
		if signal_probability>threshold and sample_y[i][0]==0 :
			false_signal += 1
			filtered_mass.append(sample_x[i][3])
		if signal_probability<=threshold and sample_y[i][0]==1 :
			false_background += 1
		if signal_probability<=threshold and sample_y[i][0]==0 :
			true_background += 1
		i += 1

	return true_signal, false_signal, false_background, true_background, filtered_mass

def train_neural_network(x, layer_sizes):

	# print("train_x sample : ",train_x[0])
	# print("test_x sample : ",test_x[0])

	prediction = neural_network_model(x,layer_sizes)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(max_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
															  y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			# print('Epoch', epoch+1, 'completed out of',max_epochs,'loss:',epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Test Accuracy:',accuracy.eval({x:test_x, y:test_y}))

		i = 0
		roc = (np.zeros(divisions+1),np.zeros(divisions+1))
		threshold_plot = (np.zeros(divisions+1),np.zeros(divisions+1))
		filtered_mass_ret = []
		
		while i/divisions <= 1:
			true_signal, false_signal, false_background, true_background, filtered_mass = confusion_matrix(tf.nn.softmax(prediction).eval(feed_dict = {x:test_x}, session = sess), test_x, test_y, i/divisions)
			filtered_mass_ret.append(filtered_mass)
			background_rejection = true_background/(true_background+false_signal)
			signal_efficiency = true_signal/(true_signal+false_background)
			
			roc[0][i] = signal_efficiency
			roc[1][i] = background_rejection
			threshold_plot[0][i] = i/divisions
			threshold_plot[1][i] = true_signal/math.sqrt(false_signal+1)
			i += 1
			
	print("Signal Efficiency: ", roc[0])
	print("Background Rejection: ", roc[1])
	auc = np.trapz(roc[1],roc[0])

	return roc, -auc, threshold_plot, filtered_mass_ret


def structure_test():

	all_nodes = []

	for tries in range (int(sys.argv[2])):
		n_nodes_hl1 = random.randint(50,300)
		n_nodes_hl2 = random.randint(100,500)
		n_nodes_hl3 = random.randint(50,300)
		print()
		print("Structure",tries+1,": ", n_nodes_hl1,n_nodes_hl2,n_nodes_hl3)
		roc, auc, threshold_plot, filtered_mass = train_neural_network(x,[n_nodes_hl1,n_nodes_hl2,n_nodes_hl3])
		node = MyStruct(roc = roc, auc = auc, threshold_plot = threshold_plot, filtered_mass = filtered_mass, name = "[%d, %d, %d]" % (n_nodes_hl1,n_nodes_hl2,n_nodes_hl3))
		all_nodes.append(node)
	return all_nodes


if __name__ == '__main__':
	all_nodes = structure_test()
	with open('./output data/3-layer %s data' % sys.argv[1], 'wb') as fp:
		pickle.dump(all_nodes,fp)


