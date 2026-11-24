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
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [1.0, "sqrt", "log2"],
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 0.9, 1.0],
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2],
                    "reg_alpha": [0, 0.01, 0.1],
                    "reg_lambda": [1, 5, 10],
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7],
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"],
                },
            }
            
            
            model_report = evaluate_model(xtrain, ytrain, xtest, ytest, models,params)

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
