import os
import sys
import pandas as pd

from src.utils.utils import fn_load_object
from src.exception.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts/model.pkl')
            preprocessor_path = os.path.join('artifacts/preprocessor.pkl')

            print("Before Loading")
            model = fn_load_object(file_path=model_path)
            preprocessor = fn_load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
        
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):       
        
        self.lunch = lunch
        self.gender = gender        
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.race_ethnicity = race_ethnicity
        self.test_preparation_course = test_preparation_course
        self.parental_level_of_education = parental_level_of_education
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],                
                "lunch": [self.lunch],                
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
                "race_ethnicity": [self.race_ethnicity],
                "test_preparation_course": [self.test_preparation_course],
                "parental_level_of_education": [self.parental_level_of_education]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)      