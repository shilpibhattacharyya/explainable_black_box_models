from abc import ABC, abstractmethod

class ExplainableModel(ABC):
	
	def __init__(self):
		self.model = None

	def train(self, x_train, y_train, validation_set=None):
		pass

	def feature_importance(self):
		pass