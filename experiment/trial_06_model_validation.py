# from pathlib import Path
# from dataclasses import dataclass
# from src.LeadGen.constants import *
# from src.LeadGen.utils.commons import read_yaml, create_directories, save_json
# from src.LeadGen.logger import logger
# from src.LeadGen.exception import CustomException
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     classification_report,
#     confusion_matrix,
#     roc_curve,
#     auc,
#     precision_recall_curve,
# )
# import numpy as np  # Import numpy for array operations
# import json  # Import json for loading training metrics

# # Entity 
# @dataclass()

# class ModelValidationConfig:
#     root_dir: Path
#     test_feature_path: Path
#     test_target_path: Path
#     model_path: Path
#     class_report: Path
#     classification_report_path: Path
#     val_metric_file_name: Path
#     conf_matrix: Path
#     roc_auc_path: Path
#     pr_auc_path: Path

#     # Model parameters
#     batch_size: int
#     learning_rate: float
#     epochs: int
#     dropout_rates: dict
#     optimizer: str
#     loss_function: str
#     activation_function: str


# class ConfigurationManager:
#     def __init__(
#         self,
#         config_filepath=CONFIG_FILE_PATH,
#         params_filepath=PARAMS_FILE_PATH,
#         schema_filepath=SCHEMA_FILE_PATH,
#     ):
#         self.config = read_yaml(config_filepath)
#         self.params = read_yaml(params_filepath)
#         self.schema = read_yaml(schema_filepath)
#         create_directories([self.config.artifacts_root])

#     def get_model_validation_config(self) -> ModelValidationConfig:
#         config = self.config.model_validation  # Ensure this matches your YAML key
#         params = self.params.dnn_params

#         create_directories([config.root_dir])

#         return ModelValidationConfig(
#             root_dir=Path(config.root_dir),
#             test_feature_path=Path(config.test_feature_path),
#             test_target_path=Path(config.test_target_path),
#             model_path=Path(config.model_path.format(model_name=config.model_name, epochs=params["epochs"])),
#             class_report=Path(config.class_report),
#             classification_report_path= Path(config.classification_report_path),
#             val_metric_file_name=Path(config.val_metric_file_name),  # Ensure Path objects
#             conf_matrix=Path(config.conf_matrix),
#             roc_auc_path=Path(config.roc_auc_path),
#             pr_auc_path=Path(config.pr_auc_path),
#             batch_size=params["batch_size"],
#             learning_rate=params["learning_rate"],
#             epochs=params["epochs"],
#             dropout_rates=params["dropout_rates"],
#             optimizer=params["optimizer"],
#             loss_function=params["loss_function"],
#             activation_function=params["activation_function"]
#         )

    
# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, dropout_rates):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 16)
#         self.dropout1 = nn.Dropout(p=dropout_rates['layer_1'])
#         self.fc2 = nn.Linear(16, 8)
#         self.dropout2 = nn.Dropout(p=dropout_rates['layer_2'])
#         self.fc4 = nn.Linear(8, 1)
#         self.dropout3 = nn.Dropout(p=dropout_rates['layer_3'])
       
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.dropout3(x)
#         x = torch.sigmoid(self.fc4(x))
#         return x

# class ModelValidation:
#     def __init__(self, config: ModelValidationConfig):
#         self.config = config

#     def load_data(self):
#         X_test_tensor = torch.load(self.config.test_feature_path)
#         y_test_tensor = torch.load(self.config.test_target_path)
#         test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#         test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
#         return test_loader
    
#     def load_model(self, input_dim):
#         try:
#             model = SimpleNN(input_dim=input_dim, dropout_rates=self.config.dropout_rates)
#             model.load_state_dict(torch.load(self.config.model_path))
#             model.eval()
#             return model
#         except CustomException as e:
#             logger.error(f"An error occurred while loading model: {e}")
#         except Exception as e:
#             logger.error(f"An unexpected error occurred while loading model: {e}")
    
#     def validate(self):
#         test_loader = self.load_data()
#         input_dim = next(iter(test_loader))[0].shape[1]
#         model = self.load_model(input_dim)
        
#         criterion = nn.BCELoss()

#         test_labels = []
#         test_predictions = []
        
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = model(inputs).squeeze()
#                 test_labels.extend(labels.cpu().numpy())
#                 test_predictions.extend(outputs.cpu().numpy())

#         precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
        
#         # Ensure recall is sorted in increasing order
#         sorted_indices = np.argsort(recall)
#         recall_sorted = np.array(recall)[sorted_indices]
#         precision_sorted = np.array(precision)[sorted_indices]
        
#         roc_pr_metrics = {
#             "roc_auc": roc_auc_score(test_labels, test_predictions),
#             "pr_auc": auc(recall_sorted, precision_sorted)
#         }

#         save_json(str(self.config.val_metric_file_name), roc_pr_metrics)  # Convert Path to str here


#         # Calculate evaluation metrics
#         accuracy = accuracy_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
#         precision = precision_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
#         recall = recall_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
#         f1 = f1_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
#         auc_roc = roc_auc_score(test_labels, test_predictions)
#         auc_pr = auc(recall, precision)
        
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"ROC-AUC Score: {auc_roc:.4f}")
#         print(f"PR-AUC Score: {auc_pr:.4f}")
        
#         # Save evaluation metrics to JSON file
#         with open(self.config.classification_report_path, 'w') as f:
#             f.write(json.dumps({
#                 'accuracy': accuracy,
#                 'precision': precision,
#                'recall': recall,
#                 'f1_score': f1,
#                 'roc_auc_score': auc_roc,
#                 'pr_auc_score': auc_pr
#             }))

        

#         self.generate_reports(test_labels, test_predictions)

#     def generate_reports(self, test_labels, test_predictions):
#         try:

#             print("Classification Report:\n", classification_report(test_labels, (np.array(test_predictions) > 0.5).astype(int)))
#             # Classification report
#             classification_repo = classification_report(test_labels, (np.array(test_predictions) > 0.5).astype(int), output_dict=True)
#             with open(self.config.class_report, 'w') as f:
#                 f.write(str(classification_repo))

#             print("Classification Report:\n", classification_repo)

#             # Confusion matrix
#             cm = confusion_matrix(test_labels, (np.array(test_predictions) > 0.5).astype(int))
#             with open(self.config.conf_matrix, "w") as f:
#                 f.write(str(cm))
            
#             plt.figure(figsize=(10, 7))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#             plt.xlabel('Predicted')
#             plt.ylabel('Actual')
#             plt.title('Test Set Confusion Matrix')
#             plt.savefig(self.config.conf_matrix)
#             plt.show()
#             plt.close()

#             # ROC-AUC
#             roc_auc = roc_auc_score(test_labels, test_predictions)
#             fpr, tpr, _ = roc_curve(test_labels, test_predictions)
#             plt.figure()
#             plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
#             plt.plot([0, 1], [0, 1], "k--")
#             plt.xlabel("False Positive Rate")
#             plt.ylabel("True Positive Rate")
#             plt.title("Test Set ROC Curve")
#             plt.legend()
#             plt.savefig(self.config.roc_auc_path)
#             plt.close()

#             # Precision-Recall AUC
#             precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
#             pr_auc = auc(recall, precision)
#             plt.figure()
#             plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
#             plt.xlabel('Recall')
#             plt.ylabel('Precision')
#             plt.title('Test Set Precision-Recall Curve')
#             plt.legend()
#             plt.savefig(self.config.pr_auc_path)
#             plt.close()

#         except CustomException as e:
#             logger.error(f"An error occurred while generating reports: {e}")
#         except Exception as e:
#             logger.error(f"An unexpected error occurred while generating reports: {e}")

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         model_validation_config = config_manager.get_model_validation_config()
#         model_validator = ModelValidation(config=model_validation_config)
#         model_validator.validate()
#     except Exception as e:
#         raise CustomException(e, sys)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import numpy as np  # Import numpy for array operations
import json  # Import json for loading training metrics
from pathlib import Path
from dataclasses import dataclass
from src.LeadGen.constants import *
from src.LeadGen.utils.commons import read_yaml, create_directories, save_json
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException


@dataclass
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_target_path: Path
    model_path: Path
    class_report: Path
    classification_report_path: Path
    val_metric_file_name: Path
    conf_matrix: Path
    roc_auc_path: Path
    pr_auc_path: Path

    # Model parameters
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict
    optimizer: str
    loss_function: str
    activation_function: str


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_model_validation_config(self) -> ModelValidationConfig:
        config = self.config.model_validation  # Ensure this matches your YAML key
        params = self.params.dnn_params

        create_directories([config.root_dir])

        return ModelValidationConfig(
            root_dir=Path(config.root_dir),
            test_feature_path=Path(config.test_feature_path),
            test_target_path=Path(config.test_target_path),
            model_path=Path(config.model_path.format(model_name=config.model_name, epochs=params["epochs"])),
            class_report=Path(config.class_report),
            classification_report_path=Path(config.classification_report_path),
            val_metric_file_name=Path(config.val_metric_file_name),  # Ensure Path objects
            conf_matrix=Path(config.conf_matrix),
            roc_auc_path=Path(config.roc_auc_path),
            pr_auc_path=Path(config.pr_auc_path),
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            dropout_rates=params["dropout_rates"],
            optimizer=params["optimizer"],
            loss_function=params["loss_function"],
            activation_function=params["activation_function"]
        )


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


class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config

    def load_data(self):
        X_test_tensor = torch.load(self.config.test_feature_path)
        y_test_tensor = torch.load(self.config.test_target_path)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return test_loader

    def load_model(self, input_dim):
        try:
            model = SimpleNN(input_dim=input_dim, dropout_rates=self.config.dropout_rates)
            model.load_state_dict(torch.load(self.config.model_path))
            model.eval()
            return model
        except CustomException as e:
            logger.error(f"An error occurred while loading model: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading model: {e}")

    def validate(self):
        test_loader = self.load_data()
        input_dim = next(iter(test_loader))[0].shape[1]
        model = self.load_model(input_dim)

        criterion = nn.BCELoss()

        test_labels = []
        test_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(outputs.cpu().numpy())



        

        # Ensure predictions are thresholded at 0.5
        predicted_classes = (np.array(test_predictions) > 0.5).astype(int)

        # Print out a few examples to ensure correctness
        print("Sample predictions: ", predicted_classes[:10])
        print("Sample true labels: ", test_labels[:10])
        
        # Classification report
        print("Classification Report:\n", classification_report(test_labels, predicted_classes))
        
        classification_repo = classification_report(test_labels, predicted_classes, output_dict=True)
        with open(self.config.class_report, 'w') as f:
            f.write(str(classification_repo))
        
        print("Classification Report:\n", classification_repo)
        
        # Other metrics
        cm = confusion_matrix(test_labels, predicted_classes)
        with open(self.config.conf_matrix, "w") as f:
            f.write(str(cm))
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Test Set Confusion Matrix')
        plt.savefig(self.config.conf_matrix)
        plt.show()
        plt.close()
        
        roc_auc = roc_auc_score(test_labels, test_predictions)
        fpr, tpr, _ = roc_curve(test_labels, test_predictions)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test Set ROC Curve")
        plt.legend()
        plt.savefig(self.config.roc_auc_path)
        plt.close()
        
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_predictions)
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Set Precision-Recall Curve')
        plt.legend()
        plt.savefig(self.config.pr_auc_path)
        plt.close()

        # Save evaluation metrics to JSON file
        with open(self.config.classification_report_path, 'w') as f:
            f.write(json.dumps({
                'accuracy': accuracy_score(test_labels, predicted_classes),
                'precision': precision_score(test_labels, predicted_classes),
                'recall': recall_score(test_labels, predicted_classes),
                'f1_score': f1_score(test_labels, predicted_classes),
                'roc_auc_score': roc_auc,
                'pr_auc_score': pr_auc
            }))



        # --------------------------------------------------------------------------------------------------------------#



        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_predictions)

        # Ensure recall is sorted in increasing order
        sorted_indices = np.argsort(recall_vals)
        recall_sorted = np.array(recall_vals)[sorted_indices]
        precision_sorted = np.array(precision_vals)[sorted_indices]

        roc_pr_metrics = {
            "roc_auc": roc_auc_score(test_labels, test_predictions),
            "pr_auc": auc(recall_sorted, precision_sorted)
        }

        save_json(str(self.config.val_metric_file_name), roc_pr_metrics)  # Convert Path to str here

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
        precision_score_val = precision_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
        recall_score_val = recall_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
        f1 = f1_score(test_labels, (np.array(test_predictions) > 0.5).astype(int))
        auc_roc = roc_auc_score(test_labels, test_predictions)
        auc_pr = auc(recall_sorted, precision_sorted)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_score_val:.4f}")
        print(f"Recall: {recall_score_val:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC Score: {auc_roc:.4f}")
        print(f"PR-AUC Score: {auc_pr:.4f}")

        # Save evaluation metrics to JSON file
        with open(self.config.classification_report_path, 'w') as f:
            f.write(json.dumps({
                'accuracy': accuracy,
                'precision': precision_score_val,
                'recall': recall_score_val,
                'f1_score': f1,
                'roc_auc_score': auc_roc,
                'pr_auc_score': auc_pr
            }))

        self.generate_reports(test_labels, test_predictions)

    def generate_reports(self, test_labels, test_predictions):
        try:
            print("Classification Report:\n", classification_report(test_labels, (np.array(test_predictions) > 0.5).astype(int)))
            # Classification report
            classification_repo = classification_report(test_labels, (np.array(test_predictions) > 0.5).astype(int), output_dict=True)
            with open(self.config.class_report, 'w') as f:
                f.write(str(classification_repo))

            print("Classification Report:\n", classification_repo)

            # Confusion matrix
            cm = confusion_matrix(test_labels, (np.array(test_predictions) > 0.5).astype(int))
            with open(self.config.conf_matrix, "w") as f:
                f.write(str(cm))

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Test Set Confusion Matrix')
            plt.savefig(self.config.conf_matrix)
            plt.show()
            plt.close()

            # ROC-AUC
            roc_auc = roc_auc_score(test_labels, test_predictions)
            fpr, tpr, _ = roc_curve(test_labels, test_predictions)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Test Set ROC Curve")
            plt.legend()
            plt.savefig(self.config.roc_auc_path)
            plt.close()

            # Precision-Recall AUC
            precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_predictions)
            pr_auc = auc(recall_vals, precision_vals)
            plt.figure()
            plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {pr_auc:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Test Set Precision-Recall Curve')
            plt.legend()
            plt.savefig(self.config.pr_auc_path)
            plt.close()

        except CustomException as e:
            logger.error(f"An error occurred while generating reports: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while generating reports: {e}")


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_validation_config = config_manager.get_model_validation_config()
        model_validator = ModelValidation(config=model_validation_config)
        model_validator.validate()
    except Exception as e:
        raise CustomException(e, sys)
