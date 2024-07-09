import pandas as pd
from category_encoders import TargetEncoder
import torch
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import save_object
from src.LeadGen.constants import *
from src.LeadGen.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_obj(self, X_train, y_train):
        try:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])

            categorical_transformer = Pipeline([
                ('target_encoder', TargetEncoder(cols=self.config.categorical_cols))
            ])

            preprocessor = ColumnTransformer([
                ('numerical', numeric_transformer, self.config.numerical_cols),
                ('categorical', categorical_transformer, self.config.categorical_cols)
            ], remainder='passthrough')

            preprocessor.fit(X_train, y_train)
            return preprocessor

        except Exception as e:
            logger.exception("Failed to create transformer object.")
            raise CustomException(f"Error creating transformer object: {e}")
        
    def save_tensors(self, tensors, names):
        for tensor, name in zip(tensors, names):
            torch.save(tensor, self.config.root_dir / f"{name}.pt")

    def train_val_test_split(self):
        try:
            logger.info("Data Splitting process has started")

            df = pd.read_csv(self.config.data_path)
            X = df.drop(columns=["Converted"])
            y = df["Converted"]

            logger.info("Splitting data into training, validation, and testing sets")
            X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

            logger.info("Converting labels to tensors")
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


            logger.info("Data Splitting process has completed")
            self.save_tensors([y_train_tensor, y_val_tensor, y_test_tensor], ["y_train_tensor", "y_val_tensor", "y_test_tensor"])

            return X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor

        except Exception as e:
            logger.exception("Failed during data splitting.")
            raise CustomException(f"Error during data splitting: {e}")
    

    def initiate_data_transformation(self, X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor):
        try:
            logger.info("Data Transformation process has started")

            preprocessor = self.get_transformer_obj(X_train, y_train_tensor)

            X_train_tensor = torch.tensor(preprocessor.transform(X_train), dtype=torch.float32)
            X_val_tensor = torch.tensor(preprocessor.transform(X_val), dtype=torch.float32)
            X_test_tensor = torch.tensor(preprocessor.transform(X_test), dtype=torch.float32)

            self.save_tensors([X_train_tensor, X_val_tensor, X_test_tensor], ["X_train_tensor", "X_val_tensor", "X_test_tensor"])

            preprocessor_path = self.config.root_dir / "preprocessor_obj.joblib"
            save_object(preprocessor_path, preprocessor)

            logger.info("Data Transformation process has completed")
            return X_train_tensor, X_val_tensor, X_test_tensor, preprocessor_path

        except Exception as e:
            logger.exception("Failed during data transformation.")
            raise CustomException(f"Error during data transformation: {e}")