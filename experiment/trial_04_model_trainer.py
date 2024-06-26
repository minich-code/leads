from dataclasses import dataclass
from pathlib import Path
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.LeadGen.logger import logger  # Replace with your custom logger
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import save_object, read_yaml, create_directories
from src.LeadGen.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from experiment.trial_03_data_transformation import DataTransformationConfig, DataTransformation


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    train_features_path: Path
    train_targets_path: Path

    # Model parameters
    batch_size: int
    learning_rate: float
    epochs: int 
    dropout_rates: float
    optimizer: str
    loss_function: str
    activation_function: str
   
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
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.dnn_params
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            train_features_path=config.train_features_path,
            train_targets_path=config.train_targets_path,
            # Model parameters
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            dropout_rates=params['dropout_rates'],
            optimizer=params['optimizer'],
            loss_function=params['loss_function'],
            activation_function=params['activation_function'],
        )
        return model_trainer_config
        

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
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

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        try:
            X_train_tensor = torch.load(self.config.train_features_path)
            y_train_tensor = torch.load(self.config.train_targets_path)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            return train_loader
        except Exception as e:
            logger.exception("Failed during load data preparation.")
            raise CustomException(f"Error during load data: {e}")


    def train_model(self):
        try:
            train_loader = self.load_data()
            input_dim = next(iter(train_loader))[0].shape[1]
            model = SimpleNN(input_dim, self.config.dropout_rates)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            for epoch in range(self.config.epochs):
                model.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    predicted_train = outputs.round()
                    correct_train += (predicted_train == labels).sum().item()
                    total_train += labels.size(0)

                train_loss = running_loss / len(train_loader)
                train_accuracy = correct_train / total_train

                logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

                # Save the model after each epoch (or at intervals)
                model_path = os.path.join(self.config.root_dir, self.config.model_name + f"_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), model_path)  # Save only the model's state dict
                
                # # Save the model
                # model_path = os.path.join(self.config.root_dir, self.config.model_name + ".pt")
                # torch.save(model, model_path) # Save the entire model
                


            logger.info(f"Model saved to {model_path}")
            logger.info("Neural Network Training process has completed")

        except Exception as e:
            logger.exception("Failed during model training.")
            raise CustomException(f"Error during model training: {e}")


    
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()

        # Instantiate the model trainer with the configuration
        model_trainer = ModelTrainer(config=model_trainer_config)

        # Train the model
        model_trainer.train_model()

    except CustomException as e:
        logger.error(f"Model trainer process failed: {e}")
