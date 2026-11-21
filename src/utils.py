

import os
import pickle as pkl


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor



def save_object(path, obj):
  dir = os.path.dirname(path)  
  os.makedirs(dir,exist_ok = True)
  pkl.dump(obj, open(path, "wb"))


def evaluate_model(xtrain, ytrain, xtest, ytest, models):

  model_report = {}

  for idx in range(len(list(models))):
    model = list(models.values())[idx]

    model.fit(xtrain, ytrain)

    y_predict = model.predict(xtest)

    model_test_score = r2_score(ytest, y_predict)

    model_report[list(models.keys())[idx]] = model_test_score

  return model_report
