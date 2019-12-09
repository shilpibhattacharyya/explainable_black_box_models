from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from ExplainableModel import ExplainableModel

default_classifier = Sequential()
default_classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=89))
default_classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
default_classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

default_batch_size = 10
default_num_epochs = 10


class KerasModel(ExplainableModel):

	def __init__(self, classifier=None, 
		batch_size=None, num_epochs=None):

		super().__init__()

		if classifier is None:
			self.classifier = default_classifier
		else:
			self.classifier = classifier

		if batch_size is None:
			self.batch_size = default_batch_size
		else:
			self.batch_size = batch_size

		if num_epochs is None:
			self.num_epochs = default_num_epochs
		else:
			self.num_epochs = num_epochs


	def set_classifier(self, classifier):
		self.classifier = classifier

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size

	def set_num_epochs(self, num_epochs):
		self.num_epochs = num_epochs


	def train(self, x_train, y_train, validation_set=None):
		classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
		classifier.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs)

		model = classifier

	def feature_importance(self):
		#placeholder
		return None
