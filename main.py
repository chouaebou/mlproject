import sys
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion



if __name__=="__main__":
    data_ingestion=DataIngestion()
    train_set,test_set=data_ingestion.initiate_data_ingestion()