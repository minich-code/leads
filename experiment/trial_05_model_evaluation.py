from pathlib import Path
from dataclasses import dataclass
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
import numpy as np
import os
import json

from src.LeadGen.constants import *
from src.LeadGen.utils.commons import read_yaml, create_directories, save_json
from src.LeadGen.logger import logger
from src.LeadGen.exception import CustomException


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    val_features_path: Path
    val_target_path: Path
    model_path: Path
    metric_file_name: Path
    validation_metrics_path: Path
    training_metrics_path: Path
    report_path: Path
    confusion_matrix_path: Path
    confusion_matrix_report: Path
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
    def __init__(self, config_filepath=MODEL_EVALUATION_CONFIG_FILEPATH, params_filepath=PARAMS_CONFIG_FILEPATH):

        self.config = read_yaml(config_filepath)

        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        params = self.params.dnn_params

        create_directories([config.root_dir])
        
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            val_features_path=Path(config.val_features_path),  
            val_target_path=Path(config.val_target_path),
            model_path=Path(config.model_path.format(model_name=config.model_name, epochs=params["epochs"])),  
            metric_file_name=Path(config.metric_file_name),  
            validation_metrics_path=Path(config.validation_metrics_path),
            training_metrics_path=Path(config.training_metrics_path),
            report_path=Path(config.report_path),  
            confusion_matrix_path=Path(config.confusion_matrix_path),  
            confusion_matrix_report=Path(config.confusion_matrix_report), 
            roc_auc_path=Path(config.roc_auc_path),  
            pr_auc_path=Path(config.pr_auc_path),  
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            dropout_rates=params["dropout_rates"],
            optimizer=params["optimizer"],
            loss_function=params["loss_function"],
            activation_function=params["activation_function"],
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
        with open(self.config.report_path, "w") as f:
            f.write(classification_rep)
        print("Classification Report:\n", classification_rep)

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
        plt.show()

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
        plt.show()

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
        plt.show()
    


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(model_eval_config)
        model_evaluator.evaluate()
    except CustomException as e:
        logger.error(f"Model trainer process failed: {e}")