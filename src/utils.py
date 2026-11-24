import os
import pickle as pkl
import sys
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score

from src.exception import CustomException

def save_object(path, obj):
  dir = os.path.dirname(path)  
  os.makedirs(dir,exist_ok = True)
  pkl.dump(obj, open(path, "wb"))


def evaluate_model(xtrain, ytrain, xtest, ytest, models, params):

  model_report = {}

  for idx in range(len(list(models))):
    model = list(models.values())[idx]
    param = params[list(models.keys())[idx]]

    gs = GridSearchCV(model, param, cv = 5)
    
    gs.fit(xtrain, ytrain)
    
    model.set_params(**gs.best_params_)
    
    model.fit(xtrain, ytrain)   

    y_predict = model.predict(xtest)

    model_test_score = r2_score(ytest, y_predict)

    model_report[list(models.keys())[idx]] = model_test_score

  return model_report

def load_object(path):
  try:
    return pkl.load(open(path, "rb"))
  except Exception as e:
    raise CustomException(e, sys)