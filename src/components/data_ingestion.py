import os
import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','text.csv')
    raw_data_path = os.path.join('artifacts','data.csv')
    

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Enteted the data ingestion method ")
        
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header = True)
            
            train_set, test_set = train_test_split(df, test_size=0.3, random_state =42)
            
            logging.info("Train test split is done")
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header = True )
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header = True)
            
                       
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    
    