import sys
from src.logging.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

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