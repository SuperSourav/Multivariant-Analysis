import tensorflow as tf

def exponential_decay_lr(train_x, batch_size, global_step):
	starter_learning_rate=0.001
	#We want to decrease the learning rate after having seen all the data 5 times
	NUM_EPOCHS_PER_DECAY=1
	LEARNING_RATE_DECAY_FACTOR=0.1
	num_batches_per_epoch=int(len(train_x)/float(batch_size))
	decay_steps=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
	learning_rate=tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,staircase=True)

	return learning_rate

def SGDR_decay_lr(train_x, batch_size, global_step):
	# Reference: https://mp.weixin.qq.com/s/QoTYg4qkiQDWQbfyy78JCg
	starter_learning_rate=0.001

	NUM_EPOCHS_PER_DECAY=1
	LEARNING_RATE_DECAY_FACTOR=0.1
	num_batches_per_epoch=int(len(train_x)/float(batch_size))
	decay_steps=int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)

	print("decay_steps = ", decay_steps)

	learning_rate = tf.train.cosine_decay_restarts(starter_learning_rate, global_step, decay_steps)

	return learning_rate