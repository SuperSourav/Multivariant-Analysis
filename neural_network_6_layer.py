from create_featuresets import create_feature_sets_and_labels
from collections import namedtuple
from neural_network_3_layer import confusion_matrix
from neural_network_3_layer import train_neural_network
import tensorflow as tf
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

with open ('./pickled data/%s_train_x' % sys.argv[1], 'rb') as fp:
    train_x_with_mass = pickle.load(fp)
    train_x = train_x_with_mass[0:len(train_x_with_mass)][0:len(train_x_with_mass)-1]

with open ('./pickled data/%s_train_y' % sys.argv[1], 'rb') as fp:
    train_y_with_mass = pickle.load(fp)
    train_y = train_y_with_mass[0:len(train_y_with_mass)][0:len(train_y_with_mass)-1]

with open ('./pickled data/%s_test_x' % sys.argv[1], 'rb') as fp:
    test_x_with_mass = pickle.load(fp)
    test_x = test_x_with_mass[0:len(test_x_with_mass)][0:len(test_x_with_mass)-1]

with open ('./pickled data/%s_test_y' % sys.argv[1], 'rb') as fp:
    test_y_with_mass = pickle.load(fp)
    test_y = test_y_with_mass[0:len(test_y_with_mass)][0:len(test_y_with_mass)-1]

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

	hidden_4_layer = {'f_fum':layer_sizes[3],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[2], layer_sizes[3]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[3]]))}

	hidden_5_layer = {'f_fum':layer_sizes[4],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[3], layer_sizes[4]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[4]]))}

	hidden_6_layer = {'f_fum':layer_sizes[5],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[4], layer_sizes[5]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[5]]))}

	output_layer = {'f_fum':None,
				'weight':tf.Variable(tf.random_normal([layer_sizes[5], n_classes])),
				'bias':tf.Variable(tf.random_normal([n_classes])),}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.sigmoid(l3)

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weight']), hidden_4_layer['bias'])
	l4 = tf.nn.sigmoid(l4)

	l5 = tf.add(tf.matmul(l4,hidden_5_layer['weight']), hidden_5_layer['bias'])
	l5 = tf.nn.sigmoid(l5)

	l6 = tf.add(tf.matmul(l5,hidden_6_layer['weight']), hidden_6_layer['bias'])
	l6 = tf.nn.sigmoid(l6)

	output = tf.matmul(l6,output_layer['weight']) + output_layer['bias']

	#output = tf.nn.softmax(output)

	return output

def plot_graph(all_nodes, data_sample):

	n = len(all_nodes)

	# bubble sort all nodes
	for i in range(n):
		for j in range(0,n-i-1):
			if all_nodes[j].auc > all_nodes[j+1].auc:
				all_nodes[j], all_nodes[j+1] = all_nodes[j+1], all_nodes[j]

	plt.figure(1)
	for i in range(n-1,0,-int(n/5)):
		roc = all_nodes[i].roc
		auc = all_nodes[i].auc
		roc_name = all_nodes[i].name
		plt.plot(roc[0],roc[1],label="%s, AUC = %f" % (roc_name,auc))
	plt.xlabel("Signal Efficiency")
	plt.ylabel("Background Rejection")
	plt.legend()
	plt.title("6-layer ROC %s data" % data_sample)
	plt.savefig("./NN results/%s data/6-layer ROC %s" % (data_sample,data_sample))

	plt.figure(2)
	for i in range(n-1,0,-int(n/5)):
		threshold_plot = all_nodes[i].threshold_plot
		roc_name = all_nodes[i].name
		plt.plot(threshold_plot[0],threshold_plot[1],label="%s" % (roc_name))
	plt.xlabel("Probability Threshold")
	plt.ylabel(r'$\frac{signal}{\sqrt{background+1}}$')
	plt.legend()
	plt.title("6-layer Probability Threshold %s data" % data_sample)
	plt.savefig("./NN results/%s data/6-layer Probability Threshold %s" % (data_sample,data_sample))

	plt.figure(3)
	max_ratio = 0
	max_index = -1
	for i in range (len(all_nodes[n-1].threshold_plot[1])):
		if all_nodes[n-1].threshold_plot[1][i] > max_ratio:
			max_ratio = all_nodes[n-1].threshold_plot[1][i]
			max_index = i
	num_bins = int(len(all_nodes[n-1].filtered_mass[max_index])/100)+1
	n, bins, patches = plt.hist(all_nodes[n-1].filtered_mass[max_index], num_bins, facecolor='blue', alpha=0.5, label="%s" % (roc_name))
	roc_name = all_nodes[len(all_nodes)-1].name
	plt.xlabel("Mass of Highest Pt Jet [GeV]")
	plt.legend()
	plt.title("Filtered Jet Mass %s data, Threshold = %f" % (data_sample, max_index/divisions))
	plt.savefig("./NN results/%s data/6-layer Filtered Jet Mass %s" % (data_sample,data_sample))


def structure_test():

	all_nodes = []

	for tries in range (int(sys.argv[2])):
		n_nodes_hl1 = random.randint(50,100)
		n_nodes_hl2 = random.randint(500,1000)
		n_nodes_hl3 = random.randint(700,1000)
		n_nodes_hl4 = random.randint(700,1000)
		n_nodes_hl5 = random.randint(500,1000)
		n_nodes_hl6 = random.randint(50,100)
		print()
		print("Structure",tries,": ", n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6)
		roc, auc, threshold_plot, filtered_mass = train_neural_network(x,[n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6])
		node = MyStruct(roc = roc, auc = auc, threshold_plot = threshold_plot, filtered_mass = filtered_mass, name = "[%d, %d, %d, %d, %d, %d]" % (n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6))
		all_nodes.append(node)
	return all_nodes


if __name__ == '__main__':
	all_nodes = structure_test()
	plot_graph(all_nodes,sys.argv[1])


