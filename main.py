import sys
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    # Get data ingestion and initiate
    data_ingestion = DataIngestion()    
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    # Get data transformed
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)