import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, Y_train):
        pass

class LinearRegressionModel(Model):

    def train(self, X_train, Y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, Y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e

