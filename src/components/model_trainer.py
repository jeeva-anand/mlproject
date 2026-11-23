import sys
import os

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainingConfig:
    model_training_path = os.path.join('artifacts', 'model.pkl')


class ModelTraining:

    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def initiate_model_training(self, train_df, test_df):
        try:

            xtrain, ytrain, xtest, ytest = (
                train_df.drop(columns=["math_score"], axis=1),
                train_df["math_score"],
                test_df.drop(columns=["math_score"], axis=1),
                test_df["math_score"]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report = evaluate_model(xtrain, ytrain, xtest, ytest, models)

            best_model_score = max(model_report.values())
            best_model_name = list(models)[list(
                model_report.values()).index(best_model_score)]

            save_object(
                self.model_training_config.model_training_path,
                models[best_model_name]
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
