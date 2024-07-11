import torch
import torch.nn as nn
import torch.nn.functional as F
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from pathlib import Path
from dataclasses import dataclass
import logging

from src.LeadGen.constants import *
from src.LeadGen.utils.commons import read_yaml, create_directories, save_json
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.entity.config_entity import ModelValidationConfig

import mlflow
import mlflow.pytorch

# Enable debug logging for MLflow
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Initialize MLflow
mlflow.set_tracking_uri("https://dagshub.com/minich-code/leads.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'minich-code'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cadc5e14617d7fae5ed8a6532906afca14f3b0f9'

# dvc remote add origin s3://dvc
# dvc remote modify origin  endpointurl https://dagshub.com/minich-code/leads.s3
# dvc remote modify origin --local access_key_id cadc5e14617d7fae5ed8a6532906afca14f3b0f9
# dvc remote modify origin --local secret_access_key cadc5e14617d7fae5ed8a6532906afca14f3b0f9



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

        test_labels, test_predictions = self._predict(model, test_loader)
        self._save_metrics(test_labels, test_predictions)
        self._plot_metrics(test_labels, test_predictions)

    def _predict(self, model, test_loader):
        test_labels = []
        test_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(outputs.cpu().numpy())
        return test_labels, test_predictions

    def _save_metrics(self, test_labels, test_predictions):
        predicted_classes = (np.array(test_predictions) > 0.5).astype(int)

        classification_repo = classification_report(test_labels, predicted_classes, output_dict=True)
        with open(self.config.class_report, 'w') as f:
            f.write(str(classification_repo))

        cm = confusion_matrix(test_labels, predicted_classes)
        # with open(self.config.conf_matrix, "w") as f:
        #     f.write(str(cm))

        roc_auc = roc_auc_score(test_labels, test_predictions)
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_predictions)
        pr_auc = auc(recall_vals, precision_vals)

        # with open(self.config.classification_report_path, 'w') as f:
        #     f.write(json.dumps({
        #         'accuracy': accuracy_score(test_labels, predicted_classes),
        #         'precision': precision_score(test_labels, predicted_classes),
        #         'recall': recall_score(test_labels, predicted_classes),
        #         'f1_score': f1_score(test_labels, predicted_classes),
        #         'roc_auc_score': roc_auc,
        #         'pr_auc_score': pr_auc
        #     }))

        sorted_indices = np.argsort(recall_vals)
        recall_sorted = np.array(recall_vals)[sorted_indices]
        precision_sorted = np.array(precision_vals)[sorted_indices]

        roc_pr_metrics = {
            "roc_auc": roc_auc,
            "pr_auc": auc(recall_sorted, precision_sorted)
        }

        save_json(str(self.config.val_metric_file_name), roc_pr_metrics)
        
        # Log metrics to MLflow
        self._log_to_mlflow(roc_pr_metrics, classification_repo, cm)

    def _plot_metrics(self, test_labels, test_predictions):
        self._plot_classification_report(test_labels, test_predictions)
        self._plot_confusion_matrix(test_labels, test_predictions)
        self._plot_roc_curve(test_labels, test_predictions)
        self._plot_pr_curve(test_labels, test_predictions)

    def _plot_classification_report(self, test_labels, test_predictions):
        predicted_classes = (np.array(test_predictions) > 0.5).astype(int)
        classification_repo = classification_report(test_labels, predicted_classes)
        print("Classification Report:\n", classification_repo)

    def _plot_confusion_matrix(self, test_labels, test_predictions):
        cm = confusion_matrix(test_labels, (np.array(test_predictions) > 0.5).astype(int))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Test Set Confusion Matrix')
        plt.savefig(os.path.join(self.config.root_dir, "val_confusion_matrix.png"))
        plt.show()
        plt.close()

    def _plot_roc_curve(self, test_labels, test_predictions):
        roc_auc = roc_auc_score(test_labels, test_predictions)
        fpr, tpr, _ = roc_curve(test_labels, test_predictions)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test Set ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "roc_auc.png"))
        plt.show()
        plt.close()

    def _plot_pr_curve(self, test_labels, test_predictions):
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_predictions)
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Set Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "pr_auc.png"))
        plt.show()
        plt.close()

    def _log_to_mlflow(self, roc_pr_metrics, classification_repo, cm):
        with mlflow.start_run() as run:
            mlflow.log_metrics({
                'roc_auc': roc_pr_metrics['roc_auc'],
                'pr_auc': roc_pr_metrics['pr_auc'],
            })
            
            # Log classification report and confusion matrix as artifacts
            classification_report_file = os.path.join(self.config.root_dir, "classification_report.json")
            confusion_matrix_file = os.path.join(self.config.root_dir, "confusion_matrix.json")

            with open(classification_report_file, 'w') as f:
                for label, metrics in classification_repo.items():
                    if isinstance(metrics, dict):
                        f.write(f"Class {label}:\n")
                        for metric, value in metrics.items():
                            f.write(f"  {metric}: {value}\n")
                    else:
                        f.write(f"{label}: {metrics}\n")
                    f.write("\n")
            
            with open(classification_report_file, 'w') as f:
                json.dump(classification_repo, f)
            
            with open(confusion_matrix_file, 'w') as f:
                json.dump(cm.tolist(), f)  # Convert numpy array to list for JSON serialization
            
            mlflow.log_artifact(classification_report_file)
            mlflow.log_artifact(confusion_matrix_file)



