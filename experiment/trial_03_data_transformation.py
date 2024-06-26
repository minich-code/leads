import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from category_encoders import TargetEncoder
import torch
from torch.utils.data import TensorDataset
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import save_object, read_yaml, create_directories
from src.LeadGen.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            numerical_cols=list(config.numerical_cols),
            categorical_cols=list(config.categorical_cols)
        )


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_obj(self, X_train, y_train):
        try:
            numerical_cols = self.config.numerical_cols
            categorical_cols = self.config.categorical_cols

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('target_encoder', TargetEncoder(cols=categorical_cols)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_transformer, numerical_cols),
                    ('categorical', categorical_transformer, categorical_cols)
                ],
                remainder='passthrough'
            )

            preprocessor.fit(X_train, y_train)
            return preprocessor

        except Exception as e:
            logger.exception("Failed to create transformer object.")
            raise CustomException(f"Error creating transformer object: {e}")

    def train_val_test_splitting(self):
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

            # Optionally save the tensors for future use
            torch.save(y_train_tensor, self.config.root_dir / "y_train_tensor.pt")
            torch.save(y_val_tensor, self.config.root_dir / "y_val_tensor.pt")
            torch.save(y_test_tensor, self.config.root_dir / "y_test_tensor.pt")

            return X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor

        except Exception as e:
            logger.exception("Failed during data splitting.")
            raise CustomException(f"Error during data splitting: {e}")

    def initiate_data_transformation(self, X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor):
        try:
            logger.info("Data Transformation process has started")

            preprocessor_obj = self.get_transformer_obj(X_train, y_train_tensor)

            X_train_transformed = preprocessor_obj.transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            logger.info("Converting features to tensors")

            X_train_tensor = torch.tensor(X_train_transformed, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_transformed, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)

            logger.info("Saving the tensors")

            # Save Tensors to .pt Files
            torch.save(X_train_tensor, self.config.root_dir / "X_train_tensor.pt")
            torch.save(X_val_tensor, self.config.root_dir / "X_val_tensor.pt")
            torch.save(X_test_tensor, self.config.root_dir / "X_test_tensor.pt")

            # Optionally save the datasets and the preprocessor
            preprocessor_path = self.config.root_dir / "preprocessor_obj.joblib"
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            logger.info("Data Transformation process has completed")
            
            # Return preprocessor path and the tensor 
            return X_train_tensor, X_val_tensor, X_test_tensor, preprocessor_path
                    

            # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            
            # torch.save(train_dataset, self.config.root_dir / "train_dataset.pt")
            # torch.save(val_dataset, self.config.root_dir / "val_dataset.pt")
            # torch.save(test_dataset, self.config.root_dir / "test_dataset.pt")

            # logger.info("Data Transformation process has completed")

            # return train_dataset, val_dataset, test_dataset, preprocessor_path

        except Exception as e:
            logger.exception("Failed during data transformation.")
            raise CustomException(f"Error during data transformation: {e}")


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor = data_transformation.train_val_test_splitting()
        train_dataset, val_dataset, test_dataset, preprocessor_path = \
            data_transformation.initiate_data_transformation(X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor)

    except CustomException as e:
        logger.error(f"Data transformation process failed: {e}")
