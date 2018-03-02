import numpy as np
import random
import pickle
import re
from collections import Counter
from itertools import zip_longest

max_lines = pow(10,3)
num_features = 27

def calculate_features_minmax(signal,background):

	min_features = [float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf"),float("inf")]
	max_features = [-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf"),-float("inf")]
	lines_count = 0

	print("Calculating features min_max")

	print("Calculating Signal")
	with open(signal,'r') as f:
		contents = f.readlines()
		for l in contents[:max_lines]:
			lines_count+=1
			for i in range (0,num_features):
				if list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i] < min_features[i]:
					min_features[i] = list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i]
				if list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i] > max_features[i]:
					max_features[i] = list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i]

	lines_count = 0

	print("Calculating Background")
	with open(background,'r') as f:
		contents = f.readlines()
		for l in contents[:max_lines]:
			lines_count+=1
			for i in range (0,num_features):
				if list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i] < min_features[i]:
					min_features[i] = list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i]
				if list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i] > max_features[i]:
					max_features[i] = list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i]
	return min_features, max_features

def sample_handling(sample,min_features,max_features,classification,down_sampling_ratio,exclusions):

	featureset = []

	lines_count = 0

	feature_indices = []

	for i in range (0,num_features):
		add = True
		for exclusion in exclusions:
			if i == exclusion:
				add = False
		if add == True:
			feature_indices.append(i)
				

	if sample=='./280_500signal.txt':
		print("feature_indices = ", feature_indices)
	
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:max_lines]:
			p = random.randint(1,100)
			counter = 0
			if p <= 100*down_sampling_ratio:
				lines_count += 1
				features = np.zeros(len(feature_indices))
				for i in feature_indices:
					# print("Feature # ",i,"added to features[",counter,"]")
					difference = max_features[i] - min_features[i]
					if difference == 0 :
						features[counter] = 1
					else:
						features[counter] = (list(map(float,re.findall("[-+]?\d+\.\d+",l)))[i] - min_features[i]) / difference
					counter += 1

				features = list(features)
				featureset.append([features,classification])

	print("%s has %d events" % (sample,lines_count))
	return featureset


def create_feature_sets_and_labels(signal,background,min_features,max_features,exclusions,test_size = 0.1):

	for exclusion in exclusions:
		if exclusion > num_features - 1:
			raise Exception('Illegal feature exclusion inputs!')
	features = []
	# print("Normalizing signal")
	features += sample_handling(signal,min_features,max_features,[1,0],1,exclusions)
	# print("Normalizing background")
	features += sample_handling(background,min_features,max_features,[0,1],1/8,exclusions)
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	print("Total = ",len(features), "events")
	print("Training = ",len(train_x), "events")
	print("Testing = ",len(test_x), "events")

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	min_features, max_features = calculate_features_minmax('./280_500signal.txt','./280_500background.txt')

	with open('./input data/max_min_features', 'wb') as fp:
		pickle.dump([min_features,max_features],fp)

	print("min_features: ",min_features)
	print("max_features: ", max_features)


	print("\nCreating low level data")
	exclusions = [3,4,5,9,10,11,15,16,17,21,22,23,24,25,26]
	low_level_train_x, low_level_train_y,low_level_test_x,low_level_test_y = create_feature_sets_and_labels('./280_500signal.txt','./280_500background.txt',min_features,max_features,exclusions)
	with open('./input data/low_level_train_x', 'wb') as fp:
		pickle.dump(low_level_train_x,fp)

	with open('./input data/low_level_train_y', 'wb') as fp:
		pickle.dump(low_level_train_y,fp)

	with open('./input data/low_level_test_x', 'wb') as fp:
		pickle.dump(low_level_test_x,fp)

	with open('./input data/low_level_test_y', 'wb') as fp:
		pickle.dump(low_level_test_y,fp)


	print("\nCreating high level data")
	exclusions = [0,1,2,6,7,8,12,13,14,18,19,20]
	high_level_train_x, high_level_train_y,high_level_test_x,high_level_test_y = create_feature_sets_and_labels('./280_500signal.txt','./280_500background.txt',min_features,max_features,exclusions)
	with open('./input data/high_level_train_x', 'wb') as fp:
		pickle.dump(high_level_train_x,fp)

	with open('./input data/high_level_train_y', 'wb') as fp:
		pickle.dump(high_level_train_y,fp)

	with open('./input data/high_level_test_x', 'wb') as fp:
		pickle.dump(high_level_test_x,fp)

	with open('./input data/high_level_test_y', 'wb') as fp:
		pickle.dump(high_level_test_y,fp)

	print("\nCreating low + high level data")
	exclusions = []
	all_train_x, all_train_y,all_test_x,all_test_y = create_feature_sets_and_labels('./280_500signal.txt','./280_500background.txt',min_features,max_features,exclusions)
	with open('./input data/all_train_x', 'wb') as fp:
		pickle.dump(all_train_x,fp)

	with open('./input data/all_train_y', 'wb') as fp:
		pickle.dump(all_train_y,fp)

	with open('./input data/all_test_x', 'wb') as fp:
		pickle.dump(all_test_x,fp)

	with open('./input data/all_test_y', 'wb') as fp:
		pickle.dump(all_test_y,fp)

	print("\nCreating no jet mass data")
	exclusions = [3,9,15]
	no_jet_mass_train_x, no_jet_mass_train_y,no_jet_mass_test_x,no_jet_mass_test_y = create_feature_sets_and_labels('./280_500signal.txt','./280_500background.txt',min_features,max_features,exclusions)
	with open('./input data/no_jet_mass_train_x', 'wb') as fp:
		pickle.dump(no_jet_mass_train_x,fp)

	with open('./input data/no_jet_mass_train_y', 'wb') as fp:
		pickle.dump(no_jet_mass_train_y,fp)

	with open('./input data/no_jet_mass_test_x', 'wb') as fp:
		pickle.dump(no_jet_mass_test_x,fp)

	with open('./input data/no_jet_mass_test_y', 'wb') as fp:
		pickle.dump(no_jet_mass_test_y,fp)

	print("\nCreating no D2 data")
	exclusions = [4,10,16]
	no_D2_train_x, no_D2_train_y,no_D2_test_x,no_D2_test_y = create_feature_sets_and_labels('./280_500signal.txt','./280_500background.txt',min_features,max_features,exclusions)
	with open('./input data/no_D2_train_x', 'wb') as fp:
		pickle.dump(no_D2_train_x,fp)

	with open('./input data/no_D2_train_y', 'wb') as fp:
		pickle.dump(no_D2_train_y,fp)

	with open('./input data/no_D2_test_x', 'wb') as fp:
		pickle.dump(no_D2_test_x,fp)

	with open('./input data/no_D2_test_y', 'wb') as fp:
		pickle.dump(no_D2_test_y,fp)
# Raw data features:
# 0->5: pt_1,eta_1,phi_1,m_1,D2_1,subjets_1,
# 6->11: pt_2,eta_2,phi_2,m_2,D2_2,subjets_2, 
# 12->17: pt_3,eta_3,phi_3,m_3,D2_3,subjets_3,
# 18->20: pt_photon.Pt()/1000,pt_photon.Eta(),pt_photon.Phi(),
# 21->27: ECF2_1,ECF3_1,ECF2_2,ECF3_2,ECF2_3,ECF3_3,eventInfo->mcEventWeight()





