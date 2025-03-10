import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logging.logger import logging
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("raw_files", "data.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")  
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info('Read the dataset as dataframe.')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)            
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 
                       
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Declaration of variables
    model_trainer = ModelTrainer()
    data_ingestion = DataIngestion() 
    data_transformation = DataTransformation()
    
    # Get data ingestion and initiate
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Get data transformed    
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
    
    # Get model trainer
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))