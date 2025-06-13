import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> RegressorMixin:
    """
    train a model on data
    Args:
        df: pandas Dataframe of the data
    """
    try:
        mlflow.sklearn.autolog()
        model = None
        model = LinearRegressionModel()
        trained_model = model.train(X_train, Y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise e