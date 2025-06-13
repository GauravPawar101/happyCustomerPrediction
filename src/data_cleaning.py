import logging
import pandas as pd
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    
    """
    Abstract class defining strategy for handling dta
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreprocessStrategy(DataStrategy):
    
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop([
                "order_id",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_purchase_timestamp",
            ],
            axis = 1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis = 1)
            return data

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis = 1)
            Y = data["review_score"]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error("Error in diving data: {e}")
            raise e

class DataCleaning:
    """
    Clean data which process and divide it
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {e}")
            raise e

if __name__ == '__main__':
    data = pd.read_csv("")
    data_cleaning = DataCleaning(data, DataPreProcessStrategy)
    data_cleaning.handle_data()