import os, sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting into train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            model_report:dict = evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException('Best model not found')
            logging.info("Best model found on training and testing dataset")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
                )
            
            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            print(f"Model used : {best_model}")
            return acc_score



        except Exception as e:
            raise CustomException(e, sys)
