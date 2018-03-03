from collections import namedtuple
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

MyStruct = namedtuple("MyStruct", "roc auc threshold_plot filtered_mass name")

def plot_graph(data_sample, num_layers):

	if num_layers!=3 and num_layers!=6 and num_layers!=10:
		raise Exception('Illegal num_layers input!')

	if data_sample!="all" and data_sample!="high_level" and data_sample!="low_level" and data_sample!="no_D2" and data_sample!="no_jet_mass":
		raise Exception('Illegal data_sample input!')

	with open ('./input data/%s_test_x' % data_sample, 'rb') as fp:
	    test_x = pickle.load(fp)

	with open('./output data/%d-layer %s data' % (num_layers,data_sample), 'rb') as fp:
		all_nodes = pickle.load(fp)

	with open('./input data/max_min_features', 'rb') as fp:
		temp = pickle.load(fp)
		min_features = temp[0]
		max_features = temp[1]

	divisions = 20

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
	plt.title("%d-layer ROC %s data" % (num_layers,data_sample))
	plt.savefig("./NN results visualizations/%s data/%d-layer ROC" % (data_sample,num_layers))
	plt.close(1)

	plt.figure(2)
	for i in range(n-1,0,-int(n/5)):
		threshold_plot = all_nodes[i].threshold_plot
		roc_name = all_nodes[i].name
		plt.plot(threshold_plot[0],threshold_plot[1],label="%s" % (roc_name))
	plt.xlabel("Probability Threshold")
	plt.ylabel(r'$\frac{signal}{\sqrt{background+1}}$')
	plt.legend()
	plt.title("%d-layer Probability Threshold %s data" % (num_layers,data_sample))
	plt.savefig("./NN results visualizations/%s data/%d-layer Probability Threshold" % (data_sample,num_layers))
	plt.close(2)

	plt.figure(3)
	max_ratio = -1
	max_index = -1
	for i in range (len(all_nodes[n-1].threshold_plot[1])):
		if all_nodes[n-1].threshold_plot[1][i] > max_ratio:
			max_ratio = all_nodes[n-1].threshold_plot[1][i]
			max_index = i
	num_bins = 100
	roc_name = all_nodes[n-1].name
	plt.hist(all_nodes[n-1].filtered_mass[max_index], num_bins, facecolor='blue', alpha=0.5, label="%s" % (roc_name))
	plt.xlabel("Mass of Highest Pt Jet [GeV]")
	plt.legend()
	plt.title("Filtered Jet Mass %s data, Threshold = %f" % (data_sample, max_index/divisions))
	plt.savefig("./NN results visualizations/%s data/%d-layer Filtered Jet Mass" % (data_sample,num_layers))
	plt.close(3)

	if num_layers == 3:
		mass_index = -1
		if data_sample == "all":
			mass_index = 3
		if data_sample == "high_level":
			mass_index = 0
		if data_sample == "no_D2":
			mass_index = 3

		plt.figure(4)
		num_bins = 100
		masses = []
		for i in range (len(test_x)):
			masses.append((test_x[i][mass_index]*(max_features[3]-min_features[3])+min_features[3])/1000)
		plt.hist(masses, num_bins, facecolor='blue', alpha=0.5)
		plt.xlabel("Mass of Highest Pt Jet [GeV]")
		plt.title("Unfiltered Jet Mass %s data" % data_sample)
		plt.savefig("./NN results visualizations/Unfiltered Jet Mass %s" % data_sample)
		plt.close(4)

# plot_graph(sys.argv[1],int(sys.argv[2]))
for data_sample in ["all","high_level","no_D2"]:
	for num_layers in [3,6,10]:
		plot_graph(data_sample,num_layers)


