
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import numpy as np
import os
import logging
import pandas as pd

from src.LeadGen.constants import *
from src.LeadGen.utils.commons import save_json
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException
from src.LeadGen.entity.config_entity import ModelEvaluationConfig

import mlflow
import mlflow.pytorch

# Enable debug logging for MLflow
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Initialize MLflow
mlflow.set_tracking_uri("https://dagshub.com/minich-code/leads.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'minich-code'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cadc5e14617d7fae5ed8a6532906afca14f3b0f9'

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


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data(self):
        X_val_tensor = torch.load(self.config.val_features_path)
        y_val_tensor = torch.load(self.config.val_target_path)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        return val_loader
    
    def load_model(self, input_dim):
        model = SimpleNN(input_dim=input_dim, dropout_rates=self.config.dropout_rates)
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        return model
    
    def evaluate(self):
        val_loader = self.load_data()
        input_dim = next(iter(val_loader))[0].shape[1]
        model = self.load_model(input_dim)
        
        all_labels, all_predictions = self._evaluate_model(model, val_loader)
        self._save_metrics(all_labels, all_predictions)
        self._plot_metrics(all_labels, all_predictions)
        self._log_to_mlflow(all_labels, all_predictions, model)

    def _evaluate_model(self, model, val_loader):
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
        return all_labels, all_predictions
    
    def _save_metrics(self, all_labels, all_predictions):
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        sorted_indices = np.argsort(recall)
        recall_sorted = np.array(recall)[sorted_indices]
        precision_sorted = np.array(precision)[sorted_indices]
        roc_pr_metrics = {
            "roc_auc": roc_auc_score(all_labels, all_predictions),
            "pr_auc": auc(recall_sorted, precision_sorted)
        }
        save_json(str(self.config.metric_file_name), roc_pr_metrics)

    def _plot_metrics(self, all_labels, all_predictions):
        self._plot_classification_report(all_labels, all_predictions)
        self._plot_confusion_matrix(all_labels, all_predictions)
        self._plot_roc_curve(all_labels, all_predictions)
        self._plot_pr_curve(all_labels, all_predictions)

    def _plot_classification_report(self, all_labels, all_predictions):
        classification_rep = classification_report(all_labels, (np.array(all_predictions) > 0.5).astype(int))
        # Save classification report as a txt file
        with open(self.config.report_path, "w") as f:
            f.write(classification_rep)
        print("Classification Report:\n", classification_rep)

        # Convert the report to a DataFrame for easier plotting
        report = classification_report(all_labels, (np.array(all_predictions) > 0.5).astype(int), output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Plot the classification report as a heatmap
        plt.figure(figsize=(10, len(report_df) / 2))  # Adjust figure size as needed
        sns.heatmap(report_df.iloc[:-2, :].T,  # Exclude 'accuracy' and 'weighted avg'
                    annot=True, 
                    cmap='viridis', 
                    cbar=False,  # No colorbar needed for this plot
                    fmt=".2f",  # Format annotations to two decimal places
                    linewidths=0.5,  # Add thin lines between cells
                    linecolor='lightgrey')  # Color of the lines

        # Enhance the heatmap
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.title("Classification Report")
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(os.path.join(self.config.root_dir, "classification_rep.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
        
        # # Plot the classification report
        # fig, ax = plt.subplots(figsize=(10, len(report_df) / 2))
        # ax.axis('off')
        # table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center', loc='center')
        
        # # Enhance the table with some styles
        # table.auto_set_font_size(False)
        # table.set_fontsize(12)
        # table.scale(1.2, 1.2)
        
        # Save the plot as an image
        # fig.savefig(os.path.join(self.config.root_dir, "classification_rep.png"), bbox_inches='tight', dpi=300)
        # plt.close(fig)



    def _plot_confusion_matrix(self, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, (np.array(all_predictions) > 0.5).astype(int))
        with open(self.config.confusion_matrix_report, "w") as f:
            f.write(str(cm))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.config.root_dir, "confusion_matrix.png"))
        plt.close()  # Close the figure to avoid display in some environments

    def _plot_roc_curve(self, all_labels, all_predictions):
        roc_auc = roc_auc_score(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "roc_auc.png"))
        plt.close()  # Close the figure to avoid display in some environments

    def _plot_pr_curve(self, all_labels, all_predictions):
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "pr_auc.png"))
        plt.close()  # Close the figure to avoid display in some environments

    def _log_to_mlflow(self, all_labels, all_predictions, model):
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        sorted_indices = np.argsort(recall)
        recall_sorted = np.array(recall)[sorted_indices]
        precision_sorted = np.array(precision)[sorted_indices]
        roc_pr_metrics = {
            "roc_auc": roc_auc_score(all_labels, all_predictions),
            "pr_auc": auc(recall_sorted, precision_sorted)
        }
        
        mlflow.log_params({
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.epochs,
            "dropout_rates": self.config.dropout_rates,
            "optimizer": self.config.optimizer,
            "loss_function": self.config.loss_function,
            "activation_function": self.config.activation_function
        })
        
        mlflow.log_metrics(roc_pr_metrics)
        
        mlflow.pytorch.log_model(model, "model")
        
        mlflow.log_artifact(self.config.metric_file_name)
        mlflow.log_artifact(self.config.report_path)
        mlflow.log_artifact(self.config.confusion_matrix_report)
        mlflow.log_artifact(os.path.join(self.config.root_dir, "confusion_matrix.png"))
        mlflow.log_artifact(os.path.join(self.config.root_dir, "roc_auc.png"))
        mlflow.log_artifact(os.path.join(self.config.root_dir, "pr_auc.png"))
        mlflow.log_artifact(os.path.join(self.config.root_dir, "classification_rep.txt")) 
        mlflow.log_artifact(os.path.join(self.config.root_dir, "classification_rep.png"))

