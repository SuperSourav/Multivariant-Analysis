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

with open ('./input data/%s_train_x' % sys.argv[1], 'rb') as fp:
    train_x = pickle.load(fp)

with open ('./input data/%s_train_y' % sys.argv[1], 'rb') as fp:
    train_y = pickle.load(fp)

with open ('./input data/%s_test_x' % sys.argv[1], 'rb') as fp:
    test_x = pickle.load(fp)

with open ('./input data/%s_test_y' % sys.argv[1], 'rb') as fp:
    test_y = pickle.load(fp)

with open('./input data/max_min_features', 'rb') as fp:
	temp = pickle.load(fp)
	min_features = temp[0]
	max_features = temp[1]

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

	hidden_7_layer = {'f_fum':layer_sizes[6],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[5], layer_sizes[6]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[6]]))}

	hidden_8_layer = {'f_fum':layer_sizes[7],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[6], layer_sizes[7]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[7]]))}

	hidden_9_layer = {'f_fum':layer_sizes[8],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[7], layer_sizes[8]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[8]]))}

	hidden_10_layer = {'f_fum':layer_sizes[9],
				  'weight':tf.Variable(tf.random_normal([layer_sizes[8], layer_sizes[9]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[9]]))}

	output_layer = {'f_fum':None,
				'weight':tf.Variable(tf.random_normal([layer_sizes[9], n_classes])),
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

	l7 = tf.add(tf.matmul(l6,hidden_7_layer['weight']), hidden_7_layer['bias'])
	l7 = tf.nn.sigmoid(l7)

	l8 = tf.add(tf.matmul(l7,hidden_8_layer['weight']), hidden_8_layer['bias'])
	l8 = tf.nn.sigmoid(l8)

	l9 = tf.add(tf.matmul(l8,hidden_9_layer['weight']), hidden_9_layer['bias'])
	l9 = tf.nn.sigmoid(l9)

	l10 = tf.add(tf.matmul(l9,hidden_10_layer['weight']), hidden_10_layer['bias'])
	l10 = tf.nn.sigmoid(l10)

	output = tf.matmul(l10,output_layer['weight']) + output_layer['bias']

	#output = tf.nn.softmax(output)

	return output


def structure_test():

	all_nodes = []

	for tries in range (int(sys.argv[2])):
		n_nodes_hl1 = random.randint(50,100)
		n_nodes_hl2 = random.randint(200,500)
		n_nodes_hl3 = random.randint(700,1000)
		n_nodes_hl4 = random.randint(700,1000)
		n_nodes_hl5 = random.randint(800,1000)
		n_nodes_hl6 = random.randint(700,1000)
		n_nodes_hl7 = random.randint(700,1000)
		n_nodes_hl8 = random.randint(500,1000)
		n_nodes_hl9 = random.randint(200,500)
		n_nodes_hl10 = random.randint(50,100)
		print()
		print("Structure",tries+1,": ", n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6,n_nodes_hl7,n_nodes_hl8,n_nodes_hl9,n_nodes_hl10)
		roc, auc, threshold_plot, filtered_mass = train_neural_network(x,[n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6,n_nodes_hl7,n_nodes_hl8,n_nodes_hl9,n_nodes_hl10])
		node = MyStruct(roc = roc, auc = auc, threshold_plot = threshold_plot, filtered_mass = filtered_mass, name = "[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]" % (n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_nodes_hl4,n_nodes_hl5,n_nodes_hl6,n_nodes_hl7,n_nodes_hl8,n_nodes_hl9,n_nodes_hl10))
		all_nodes.append(node)
	return all_nodes

if __name__ == '__main__':
	all_nodes = structure_test()
	with open('./output data/10-layer %s data' % sys.argv[1], 'wb') as fp:
		pickle.dump(all_nodes,fp)


