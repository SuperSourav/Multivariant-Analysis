from collections import namedtuple
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import os
import glob

MyStruct = namedtuple("MyStruct", "roc auc threshold_plot filtered_mass epoch_losses name")

def getInfo(data_sample, num_layers, lr_model):
	if data_sample!="all" and data_sample!="high_level" and data_sample!="no_D2":
		raise Exception('Illegal data_sample input!')

	with open ('./input data/%s_test_x' % data_sample, 'rb') as fp:
		test_x = pickle.load(fp)

	with open('./output_data/%d-layer %s data %s_lr' % (num_layers,data_sample, lr_model), 'rb') as fp:
		all_nodes = pickle.load(fp)

	n = len(all_nodes)

	# bubble sort all nodes in increasing auc
	for i in range(n):
		for j in range(0,n-i-1):
			if all_nodes[j].auc > all_nodes[j+1].auc:
				all_nodes[j], all_nodes[j+1] = all_nodes[j+1], all_nodes[j]

	max_ratio = -1
	for i in range (len(all_nodes[n-1].threshold_plot[1])):
		if all_nodes[n-1].threshold_plot[1][i] > max_ratio:
			max_ratio = all_nodes[n-1].threshold_plot[1][i]

	length = len(all_nodes[n-1].epoch_losses)

	return all_nodes[n-1].auc, max_ratio, all_nodes[n-1].epoch_losses[length-1], length

def make_dictionary():
	dictionary = {}
	categories = ["all", "high_level", "no_D2"]
	metrics = ["auc", "sb_ratio", "loss", "epochs"]
	lr_models = ["constant", "staircase", "triangular", "exp", "SGDR"]
	for category in categories:
		dictionary[category] = {}
		for metric in metrics:
			dictionary[category][metric] = {}
			for lr_model in lr_models:
				plot = []
				plot.append([])
				plot.append([])
				dictionary[category][metric][lr_model] = plot

	path = './output_data'

	for filename in glob.glob(os.path.join(path, '*lr')):
		filename = re.split("/",filename)[2]
		num_layers = re.split("-",filename)[0]
		data_sample = re.split(" ",filename)[1]
		lr_model = re.split("_",re.split(" ",filename)[3])[0]

		auc, sb_ratio, loss, epochs = getInfo(data_sample, int(num_layers),lr_model)

		dictionary[data_sample]['auc'][lr_model][0].append(int(num_layers))
		dictionary[data_sample]['auc'][lr_model][1].append(auc)

		dictionary[data_sample]['sb_ratio'][lr_model][0].append(int(num_layers))
		dictionary[data_sample]['sb_ratio'][lr_model][1].append(sb_ratio)

		dictionary[data_sample]['loss'][lr_model][0].append(int(num_layers))
		dictionary[data_sample]['loss'][lr_model][1].append(loss)

		dictionary[data_sample]['epochs'][lr_model][0].append(int(num_layers))
		dictionary[data_sample]['epochs'][lr_model][1].append(epochs)

	return dictionary

def plot_graph(dictionary):
	categories = ["all", "high_level", "no_D2"]
	metrics = ["auc", "sb_ratio", "loss", "epochs"]
	lr_models = ["constant", "staircase", "triangular", "exp", "SGDR"]

	for category in categories:
		for metric in metrics:
			plt.figure(1)
			for lr_model in lr_models:
				x = dictionary[category][metric][lr_model][0]
				y = dictionary[category][metric][lr_model][1]
				temp = [(a, b) for a, b in zip(x, y)]
				temp = sorted(temp, key=lambda x: x[0])
				x = [i[0] for i in temp]
				y = [i[1] for i in temp]
				plt.plot(x,y,label="%s" % lr_model)
			plt.xlabel("Number of layers")
			plt.ylabel("%s" % metric)
			plt.legend()
			plt.title("Evolution of %s, %s data" % (metric, category))
			plt.savefig("./NN_results_visualizations/changing_layers/Evolution_of_%s_%s_data" % (metric, category))
			plt.close(1)

if __name__ == '__main__':
	dictionary = make_dictionary()
	plot_graph(dictionary)





			

