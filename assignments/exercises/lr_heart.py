import sys
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split


class DataSet():

	def __init__(self, file_name, percent):
		self.file_name = file_name
		self.percent = percent
		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []

	def generate_train_test(self):
		line_cnt = 0
		data = []
		with open(self.file_name, 'r') as filetoread:
			for line in filetoread:
				line_cnt += 1
				if line_cnt == 1:
					continue
				linetuple = line.strip().split(',')
				data.append(linetuple)
		data_matrix = np.array(data)

		# convert famhist feature into either 0 or 1
		conv_col = data_matrix[:,4]
		le = preprocessing.LabelEncoder()
		le_col = le.fit_transform(conv_col)

		# for the label column, use one-hot encoder
		label_col = np.array([data_matrix[:,9]]).T
		enc = OneHotEncoder()
		label_enc = enc.fit_transform(label_col).toarray()

		# combine all the processed features together
		# keep processed label (one hot encoder) and features separately
		processed_feature = np.concatenate((data_matrix[:, range(0,4)], \
											np.array([le_col]).T, \
											data_matrix[:, range(5, 9)]), axis = 1).\
												astype(np.float)
		# perform z-normalization on features
		processed_feature_norm = preprocessing.normalize(processed_feature,
														 norm = 'l2')

		self.x_train, self.x_test, self.y_train, self.y_test = \
			train_test_split(processed_feature_norm, label_enc,
							 test_size = self.percent)


def build_lr():
	x = tf.placeholder(tf.float32, [None, 9])
	y = tf.placeholder(tf.float32, [None, 2])

	# declare the weights and bias connecting from input to hidden layer
	W = tf.Variable(tf.random_normal([9, 2], stddev = 0.01), name = 'W')
	b = tf.Variable(tf.random_normal([2]), name = 'b')

	# calculate the output layer
	output_layer = tf.add(tf.matmul(x, W), b)
	y_ = tf.nn.softmax(output_layer)
	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, \
	#														labels = y)
	#loss = tf.reduce_mean(cross_entropy)
	loss = tf.nn.l2_loss(y_ - y)
	# add an optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001). \
												minimize(loss)
	correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	return loss, optimizer, accuracy, x, y


def run_lr(dataset, batch_size):
	n_epochs = 100
	num_train = dataset.x_train.shape[0]
	total_batch = num_train / batch_size + 1
	with tf.Session() as sess:
		loss, optimizer, accuracy, x, y = build_lr()
		sess.run(tf.initialize_all_variables())
		for i in xrange(n_epochs):
			avg_cost = 0.0
			for k in xrange(0, num_train, batch_size):
				if k + batch_size > num_train:
					batch_x = dataset.x_train[range(i, num_train),:]
					batch_y = dataset.y_train[range(i, num_train),:]
				else:
					batch_x = dataset.x_train[range(i, i + batch_size),:]
					batch_y = dataset.y_train[range(i, i + batch_size),:]

				opt, c = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
				avg_cost += c

			test_acc = sess.run(accuracy, \
							    feed_dict = {x: dataset.x_test, y: dataset.y_test})
			print 'training loss is ' + str(float(avg_cost) / total_batch)
			print 'test accuracy is ' + str(test_acc)
			print


if __name__ == '__main__':

	file_name = str(sys.argv[1])
	percent = float(sys.argv[2])
	dataset = DataSet(file_name, percent)
	dataset.generate_train_test()

	batch_size = int(sys.argv[3])
	run_lr(dataset, batch_size)
