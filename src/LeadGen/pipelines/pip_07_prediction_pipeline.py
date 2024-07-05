import numpy as np 
import pandas as pd 
import os 
import sys 
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from src.LeadGen.logger import logger  # Replace with your custom logger
from src.LeadGen.exception import CustomException

from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import read_yaml, load_object
from src.LeadGen.constants import *


class SimpleNN(nn.Module):
    def __init__(self, input_dim, dropout_rates):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout1 = nn.Dropout(p=dropout_rates['layer_1'])
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(p=dropout_rates['layer_2'])
        self.fc4 = nn.Linear(8, 1)
        self.dropout3 = nn.Dropout(p=dropout_rates['layer_3'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x
    

class PredictionManager:
    def __init__(self, prediction_pipeline_config=PREDICTION_PIPELINE_FILEPATH):
        self.config = read_yaml(prediction_pipeline_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_preprocessor(self):
        preprocessor_path = load_object(self.config.preprocessor_path)

        return preprocessor_path

    def load_model(self, input_dim):
        model = SimpleNN(input_dim=input_dim, dropout_rates=self.config.dropout_rates)
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        return model
    
    def make_predictions(self, features):
        try:
            # log message 
            logger.info("Making Predictions")

            # Transform the features using preprocessor 
            preprocessor = self.load_preprocessor()
            transformed_features = preprocessor.transform(features)
            
            # Make predictions using model
            model = self.load_model(len(transformed_features[0]))
            inputs = torch.tensor(transformed_features, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predictions = model(inputs).squeeze().cpu().numpy()

            # Return the predictions
            logger.info("Predictions made successfully")
            
            return predictions
           
        except Exception as e:
            raise CustomException(e, sys)
        

# Create a class to represent the input features 
class CustomData:
    def __init__(self, **kwargs):
        # Initiate the attributes using kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Define a method to convert data object to a dataframe 
    def get_data_as_dataframe(self):
        try:
            # log message 
            logger.info("Converting data object to a dataframe")
            # Convert the data object to a dataframe 
            data_dict = {key: [getattr(self, key)] for key in vars(self)}

            # Convert the dictionary to dataframe in the return  
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException (e, sys)
        

if __name__ == "__main__":
    try:
        # Example usage
        prediction_manager = PredictionManager()
        custom_data = CustomData()  # Replace with actual features
        features_df = custom_data.get_data_as_dataframe()
        predictions = prediction_manager.make_predictions(features_df)
        print(predictions)
    except CustomException as e:
        logger.error(f"Prediction process failed: {e}")
           
