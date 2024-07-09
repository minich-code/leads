
import setuptools
import logging
from dataclasses import dataclass
from pathlib import Path
import os
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
import mlflow
import mlflow.pytorch

# Enable debug logging for MLflow
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Initialize MLflow
mlflow.set_tracking_uri("https://dagshub.com/minich-code/leads.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'minich-code'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cadc5e14617d7fae5ed8a6532906afca14f3b0f9'

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    train_features_path: Path
    train_targets_path: Path
    val_features_path: Path
    val_targets_path: Path
    val_metrics_path: Path
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict  # Updated to dict for dropout rates per layer
    optimizer: str
    loss_function: str
    activation_function: str

class ConfigurationManager:
    def __init__(self, model_training_config=MODEL_TRAINER_CONFIG_FILEPATH, params_config=PARAMS_CONFIG_FILEPATH):
        self.training_config = read_yaml(model_training_config)
        self.params = read_yaml(params_config)
        create_directories([self.training_config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.training_config.model_trainer
        params = self.params.dnn_params

        create_directories([trainer_config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(trainer_config.root_dir),
            model_name=trainer_config.model_name,
            train_features_path=trainer_config.train_features_path,
            train_targets_path=trainer_config.train_targets_path,
            val_features_path=trainer_config.val_features_path,
            val_targets_path=trainer_config.val_targets_path,
            val_metrics_path=Path(trainer_config.val_metrics_path),
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            dropout_rates=params['dropout_rates'],
            optimizer=params['optimizer'],
            loss_function=params['loss_function'],
            activation_function=params['activation_function'],
        )

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
            logger.exception("Failed during data loading.")
            raise CustomException(f"Error during data loading: {e}")

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

            mlflow.start_run()
            mlflow.log_params({
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "dropout_rates": self.config.dropout_rates,
                "optimizer": self.config.optimizer,
                "loss_function": self.config.loss_function,
                "activation_function": self.config.activation_function
            })

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

                logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}")
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}")

                # Save the model after each epoch (or at intervals)
                model_path = os.path.join(self.config.root_dir, self.config.model_name + f"_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), model_path)  # Save only the model's state dict

                # Save the model checkpoints for each epoch
                for epoch in range(self.config.epochs):
                    epoch_model_path = os.path.join(self.config.root_dir, self.config.model_name + f"_epoch_{epoch+1}.pt")
                    torch.save(model.state_dict(), epoch_model_path)  # Save only the model's state dict

               
                # Save the plots to the root_dir
                plt.savefig(os.path.join(self.config.root_dir, "training_validation_metrics.png"))

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=epoch)

                # Log the model for each epoch
                mlflow.pytorch.log_model(model, artifact_path=f"models/epoch_{epoch+1}")

            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/models/epoch_{self.config.epochs}"
            model_name = self.config.model_name
            try:
                mlflow.register_model(model_uri, model_name)
            except Exception as e:
                logger.error(f"Model registration failed: {e}")

            mlflow.end_run()

            training_metrics = {
                "training_losses": training_losses,
                "training_accuracies": training_accuracies,
                "validation_losses": validation_losses,
                "validation_accuracies": validation_accuracies,
            }

            # Save training and validation metrics to a JSON file
            save_json(self.config.val_metrics_path, training_metrics)
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

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.root_dir, "training_validation_metrics.png"))
        plt.close()
        logger.info(f"Training and validation metrics plotted and saved to {os.path.join(self.config.root_dir, 'training_validation_metrics.png')}")



if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train_model()
    except CustomException as e:
        logger.error(f"Model trainer process failed: {e}")
