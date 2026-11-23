
import pandas as pd
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
  preprocess_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
  def __init__(self):
      self.data_transformation_config = DataTransformationConfig()

  def get_data_transformer_object(self):
    try:

      logging.info("Entered the Preprocessing method")

      numerical_columns = ["writing_score", "reading_score"]
      categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]

      num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()) ]
            )
      cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
      
      preprocessor=ColumnTransformer([
          ("num_pipeline",num_pipeline,numerical_columns),
          ("cat_pipelines",cat_pipeline,categorical_columns)
      ])

      logging.info("Preprocessing has been done.....")
      
      return preprocessor
      
    except Exception as e:
      raise CustomException(e,sys)

  def initiate_data_transformation(self,train_path, test_path):
    try:

      logging.info("Read train and test data completed")
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)
    

      logging.info("Obtaining preprocessing object")
      self.preprocss_obj = self.get_data_transformer_object()

      target_feature = 'math_score'

      train_features_df = train_df.drop(columns=[target_feature], axis=1)
      train_target = train_df[target_feature]

      test_features_df = test_df.drop(columns=[target_feature], axis=1)
      test_target = test_df[target_feature]

      logging.info( f"Applying preprocessing object on training dataframe and testing dataframe.")

      train_transformed_df = self.preprocss_obj.fit_transform(train_features_df)
      test_transformed_df = self.preprocss_obj.fit_transform(test_features_df)

           
      train_df = pd.concat([pd.DataFrame(train_transformed_df), pd.DataFrame(train_target)], axis = 1)
      test_df = pd.concat([pd.DataFrame(test_transformed_df), pd.DataFrame(test_target)], axis = 1)

      logging.info(f"Saved preprocessing object.")

      save_object(self.data_transformation_config.preprocess_obj_file_path, self.preprocss_obj)
      
      return train_df, test_df
      
    except Exception as e:      
      raise CustomException(e,sys)


