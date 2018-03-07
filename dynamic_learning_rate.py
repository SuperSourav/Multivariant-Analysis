import tensorflow as tf
import numpy as np
import math

def exponential_decay_lr(train_x, batch_size, global_step):
	starter_learning_rate=0.001
	#We want to decrease the learning rate after having seen all the data 1 times
	NUM_EPOCHS_PER_DECAY=1
	LEARNING_RATE_DECAY_FACTOR=0.1
	num_batches_per_epoch=int(len(train_x)/float(batch_size))
	decay_steps=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
	learning_rate=tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=False)

	return learning_rate

def SGDR_decay_lr(train_x, batch_size, global_step,
	warmup_learning_rate=0.0,warmup_steps=0):
	NUM_EPOCHS_PER_DECAY=2
	num_batches_per_epoch=int(len(train_x)/float(batch_size))
	total_steps=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)

	learning_rate_base = 0.001

	if learning_rate_base < warmup_learning_rate:
		raise ValueError('learning_rate_base must be larger ''or equal to warmup_learning_rate.')
	if total_steps < warmup_steps:
		raise ValueError('total_steps must be larger or equal to ''warmup_steps.')
	learning_rate = 0.5 * learning_rate_base * (
	  1 + tf.cos(np.pi * (np.remainder(tf.cast(global_step, tf.float32) - warmup_steps,total_steps)
						 ) / float(total_steps - warmup_steps)))
	learning_rate = tf.where(
	np.remainder(tf.cast(global_step, tf.int32),total_steps)==0,
	learning_rate_base,
	learning_rate)

	if warmup_steps > 0:
		slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
		pre_cosine_learning_rate = slope * tf.cast(
			global_step, tf.float32) + warmup_learning_rate
		learning_rate = tf.where(
			tf.less(tf.cast(global_step, tf.int32), warmup_steps),
			pre_cosine_learning_rate,
			learning_rate)

	return learning_rate

def triangular_lr(train_x, batch_size, global_step):
	# Reference: https://mp.weixin.qq.com/s/QoTYg4qkiQDWQbfyy78JCg
	NUM_EPOCHS_PER_DECAY=1
	num_batches_per_epoch=int(len(train_x)/float(batch_size))
	half_period =int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)

	# print("half_period = ",half_period)

	max_lr = 0.001
	min_lr = max_lr/100

	slope = 1/half_period

	phase = np.remainder(tf.cast(global_step, tf.float32),2*half_period)
	perturbation = tf.where(phase<half_period, slope * phase, 1-slope*(phase-half_period))

	learning_rate = min_lr + (max_lr-min_lr) * perturbation

	return learning_rate















