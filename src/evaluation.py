import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our models
    """

    @abstractmethod
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        pass

class MSE(Evaluation):

    def calculate_scores(self, Y_true, Y_pred):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(Y_true, Y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e

class R2Score(Evaluation):

    def calculate_scores(self, Y_true, Y_pred):
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(Y_true, Y_pred)
            logging.info(f"R2 Score {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e

class RMSE(Evaluation):
    def calculate_scores(self, Y_true, Y_pred):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(Y_true, Y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e