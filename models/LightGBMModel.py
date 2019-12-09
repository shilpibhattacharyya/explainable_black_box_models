import lightgbm

from ExplainableModel import ExplainableModel

default_parameters = {
	'application': 'binary',
	'objective': 'binary',
	'metric': 'auc',
	'is_unbalance': 'true',
	'boosting': 'gbdt',
	'num_leaves': 31,
	'feature_fraction': 0.5,
	'bagging_fraction': 0.5,
	'bagging_freq': 20,
	'learning_rate': 0.05,
	'verbose': 0
}

default_num_boos_rounds = 5000
default_early_stopping_rounds = 100

class LightGBMModel(ExplainableModel):

	def __init__(self, parameters=None, num_boost_round=None,
		early_stopping_round=None, features=None):

		super().__init__()

		if parameters != None:
		    self.parameters = parameters
		else:
		    self.parameters = default_parameters

		if num_boost_round != None:
			self.num_boost_round = num_boost_round
		else:
			self.num_boost_round = default_num_boos_rounds

		if early_stopping_round != None:
			self.early_stopping_round = early_stopping_round
		else:
			self.early_stopping_round = default_early_stopping_rounds


		self.features = features


	def set_parameters(self, model_parameters):
		self.parameters = model_parameters


	def set_num_boost_rounds(self, num_rounds):
		self.num_boost_round = num_rounds


	def set_num_early_stopping_rounds(self, num_rounds):
		self.early_stopping_round = num_rounds

	def train(self, x_train, y_train, 
		validation_set=None):
		if self.features is None:
			training_data = lightgbm.Dataset(x_train, label=y_train)
		else:
			training_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=self.features)

		if validation_set is None:
			self.model = lightgbm.train(self.parameters,
	    				training_data,
	    				num_boost_round=self.num_boost_round,
	    				early_stopping_rounds=self.early_stopping_round)
		else:
			self.model = lightgbm.train(self.parameters,
	    				training_data,
	    				validation_set,
	    				num_boost_round=self.num_boost_round,
	    				early_stopping_rounds=self.early_stopping_round)

	def feature_importance(self):
		if model is not None:
			return model.feature_importance()

