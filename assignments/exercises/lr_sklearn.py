import sys
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score


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

		label_col = np.array([data_matrix[:,9]]).T.astype(np.int)
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
			train_test_split(processed_feature_norm, label_col,
							 test_size = self.percent)


def run_lr(dataset):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit_transform(dataset.x_train, dataset.y_train)
    pred_label = logreg.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, pred_label)
    return accuracy


if __name__ == '__main__':

    file_name = str(sys.argv[1])
    percent = float(sys.argv[2])

    aver_acc = 0.0
    for _ in xrange(10):
        dataset = DataSet(file_name, percent)
        dataset.generate_train_test()
        accuracy = run_lr(dataset)
        aver_acc += accuracy
        
    print str(float(aver_acc) / 10)
