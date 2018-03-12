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
max_epochs = 50
divisions = 20

x = tf.placeholder('float',[None, len(train_x[0])])	
y = tf.placeholder('float',[None, len(train_y[0])])
MyStruct = namedtuple("MyStruct", "roc auc threshold_plot filtered_mass epoch_losses name")

def neural_network_model(data, layer_sizes):
	
	# Starting layer
	hidden_layers = []
	layers = []

	hidden_layer = {'f_fum':layer_sizes[0],
				  'weight':tf.Variable(tf.random_normal([len(train_x[0]), layer_sizes[0]])),
				  'bias':tf.Variable(tf.random_normal([layer_sizes[0]]))}
	layer = tf.add(tf.matmul(data,hidden_layer['weight']), hidden_layer['bias'])
	layer = tf.nn.sigmoid(layer)

	hidden_layers.append(hidden_layer)
	layers.append(layer)

	for i in range (1,len(layer_sizes)):
		hidden_layer = {'f_fum':layer_sizes[i],
					  'weight':tf.Variable(tf.random_normal([layer_sizes[i-1], layer_sizes[i]])),
					  'bias':tf.Variable(tf.random_normal([layer_sizes[i]]))}
		hidden_layers.append(hidden_layer)
		# layer = tf.add(tf.matmul(layers[i-1],hidden_layers[i]['weight']), hidden_layers[i]['bias'])
		layer = tf.matmul(layers[i-1],hidden_layers[i]['weight'])
		batch_mean, batch_var = tf.nn.moments(layer,[0])
		scale = tf.Variable(tf.ones([layer_sizes[i]]))
		beta = tf.Variable(tf.zeros([layer_sizes[i]]))
		layer = tf.nn.batch_normalization(layer,batch_mean,batch_var,beta,scale,1e-3)
		layer = tf.nn.sigmoid(layer)
		layers.append(layer)

	output_layer = {'f_fum':None,
				'weight':tf.Variable(tf.random_normal([layer_sizes[len(layer_sizes)-1], n_classes])),
				'bias':tf.Variable(tf.random_normal([n_classes])),}

	output = tf.matmul(layers[len(layer_sizes)-1],output_layer['weight']) + output_layer['bias']

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
			filtered_mass.append((sample_x[i][3]*(max_features[3]-min_features[3])+min_features[3])/1000)
		if signal_probability>threshold and sample_y[i][0]==0 :
			false_signal += 1
			filtered_mass.append((sample_x[i][3]*(max_features[3]-min_features[3])+min_features[3])/1000)
		if signal_probability<=threshold and sample_y[i][0]==1 :
			false_background += 1
		if signal_probability<=threshold and sample_y[i][0]==0 :
			true_background += 1
		i += 1

	return true_signal, false_signal, false_background, true_background, filtered_mass

def train_neural_network(x, layer_sizes, lr_model):

	global_step=tf.Variable(0,trainable=False)

	exponential_decay_lr = dynamic_learning_rate.exponential_decay_lr(train_x, batch_size, global_step)

	triangular_lr = dynamic_learning_rate.triangular_lr(train_x, batch_size, global_step)

	SGDR_decay_lr = dynamic_learning_rate.SGDR_decay_lr(train_x, batch_size, global_step)

	staircase_decay_lr = dynamic_learning_rate.staircase_decay_lr(train_x, batch_size, global_step)

	learning_rate = tf.where(lr_model == "exp", exponential_decay_lr, 
		tf.where(lr_model == "triangular", triangular_lr, 
			tf.where(lr_model == "sgdr", SGDR_decay_lr, 
				tf.where(lr_model == "staircase", staircase_decay_lr, 0.001))))

	prediction = neural_network_model(x,layer_sizes)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step = global_step)

	config = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64, allow_soft_placement=True, device_count = {'CPU': 8})

	epoch_losses = []

	lr_val = []

	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())

		epoch = 0
		
		while (True):
			if (epoch+1>max_epochs):
				break
			epoch_loss = 0
			i=0
			# print("global_step = ", tf.train.global_step(sess, global_step))
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
															  y: batch_y})
				epoch_loss += c
				i+=batch_size

				lr_val.append([sess.run(learning_rate)])
				
			epoch_losses.append(epoch_loss)

			print('Epoch', epoch+1, 'completed out of',max_epochs,'loss:',epoch_loss)
			if (epoch>=5):
				previous_epoch_loss_average = 0
				for j in range(5):
					previous_epoch_loss_average += epoch_losses[epoch-j-1]
				previous_epoch_loss_average = previous_epoch_loss_average/5
				if (abs(epoch_loss-previous_epoch_loss_average)/previous_epoch_loss_average<1/100):
					break
			epoch += 1
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		# plt.figure()
		# plt.plot(lr_val,"-b")
		# plt.xlabel("Global Step")
		# plt.ylabel("Learning Rate")
		# plt.title("Evolution of learning rate" )
		# plt.show()

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

	return roc, -auc, threshold_plot, filtered_mass_ret, epoch_losses


def structure_test():

	all_nodes = []

	num_layers = int(sys.argv[3])

	max_num_of_nodes = len(train_x)/12/(n_classes+len(train_x[0]))
	sigma = num_layers/2.2

	for tries in range (int(sys.argv[2])):
		n_nodes = []
		for i in range (num_layers):
			scaling = max_num_of_nodes/norm.pdf(0,loc = 0, scale = sigma)
			upper = int(norm.pdf(i+1,loc = 1, scale = sigma)*scaling)
			lower = int(upper*2/3)
			# print("upper = ", upper, "lower = ",lower)
			n_nodes.append(random.randint(lower,upper))
		print()
		print("Structure",tries+1,": ", n_nodes, sys.argv[4], " learning rate")
		roc, auc, threshold_plot, filtered_mass, epoch_losses = train_neural_network(x,n_nodes, sys.argv[4])
		node = MyStruct(roc = roc, auc = auc, threshold_plot = threshold_plot, filtered_mass = filtered_mass, epoch_losses = epoch_losses, name = "[%s]" % n_nodes)
		all_nodes.append(node)
	return all_nodes


if __name__ == '__main__':
	if sys.argv[1] != "all" and sys.argv[1] != "high_level" and sys.argv[1] != "no_D2":
		raise Exception('Illegal data_sample input!')
	if int(sys.argv[3]) > 10:
		raise Exception('Neural net is too deep!')

	print("Size of training sample = ",len(train_x))
	all_nodes = structure_test()
	

	with open('./output_data/%d-layer %s data %s_lr' % (int(sys.argv[3]),sys.argv[1],sys.argv[4]), 'wb') as fp:
		pickle.dump(all_nodes,fp)


