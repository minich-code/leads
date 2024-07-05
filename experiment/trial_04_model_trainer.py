from dataclasses import dataclass
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.LeadGen.logger import logger  
from src.LeadGen.exception import CustomException
from src.LeadGen.utils.commons import save_json, read_yaml, create_directories
from src.LeadGen.constants import *



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    train_features_path: Path
    train_targets_path: Path
    val_features_path: Path
    val_targets_path: Path
    val_metrics_path: Path

    # Model parameters
    batch_size: int
    learning_rate: float
    epochs: int 
    dropout_rates: float
    optimizer: str
    loss_function: str
    activation_function: str
   
class ConfigurationManager:
    def __init__(self, config_filepath=MODEL_TRAINER_CONFIG_FILEPATH, params_filepath=PARAMS_CONFIG_FILEPATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.dnn_params
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            train_features_path=config.train_features_path,
            train_targets_path=config.train_targets_path,
            val_features_path=config.val_features_path,
            val_targets_path=config.val_targets_path,
            val_metrics_path=Path(config.val_metrics_path),

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
        self.dropout3 = nn.Dropout(p=dropout_rates['layer_3'])
        self.fc4 = nn.Linear(8, 1)
        

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
            X_val_tensor = torch.load(self.config.val_features_path)
            y_val_tensor = torch.load(self.config.val_targets_path)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

            return train_loader, val_loader
        
        except Exception as e:
            logger.exception("Failed during load data preparation.")
            raise CustomException(f"Error during load data: {e}")


    def train_model(self):
        try:
            train_loader, val_loader = self.load_data()
            input_dim = next(iter(train_loader))[0].shape[1]
            model = SimpleNN(input_dim, self.config.dropout_rates)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            training_losses = []
            training_accuracies = []
            validation_losses = []
            validation_accuracies = []

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

                training_losses.append(train_loss)
                training_accuracies.append(train_accuracy)


                # Validation phase
                model.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                all_labels = []
                all_predictions = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        predicted_val = outputs.round()
                        correct_val += (predicted_val == labels).sum().item()
                        total_val += labels.size(0)
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(outputs.cpu().numpy())

                val_loss /= len(val_loader)
                val_accuracy = correct_val / total_val

                validation_losses.append(val_loss)
                validation_accuracies.append(val_accuracy)


                # Print both validation loss and accuracy for training and validation 
                logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}")
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}")

                # Save the model after each epoch (or at intervals)
                model_path = os.path.join(self.config.root_dir, self.config.model_name + f"_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), model_path)  # Save only the model's state dict
                
                # # Save the model
                # model_path = os.path.join(self.config.root_dir, self.config.model_name + ".pt")
                # torch.save(model, model_path) # Save the entire model

                # Save the training metrics to a file
            training_metrics = {
                "training_losses": training_losses,
                "training_accuracies": training_accuracies,
                "validation_losses": validation_losses,
                "validation_accuracies": validation_accuracies,
                
            }
           
            save_json(Path(self.config.val_metrics_path), training_metrics)
        

            logger.info(f"Model saved to {model_path}")
            logger.info("Neural Network Training process has completed")

            # Plot the training and validation loss and accuracy curves
            self.plot_metrics(training_losses, validation_losses, training_accuracies, validation_accuracies)

        except Exception as e:
            logger.exception("Failed during model training.")
            raise CustomException(f"Error during model training: {e}")
    
    def plot_metrics(self, training_losses, validation_losses, training_accuracies, validation_accuracies):
        plt.figure(figsize=(12, 5))

        # Loss curves 
        plt.subplot(1, 2, 1)
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(training_accuracies, label="Training Accuracy")
        plt.plot(validation_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()

        plt.show()
        plt.savefig(os.path.join(self.config.root_dir, "training_loss_curves.png"))
   
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

        
