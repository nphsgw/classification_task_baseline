import json
import torch.utils
import torch.utils.data
import torch
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
    MulticlassAUPRC,
    MulticlassPrecisionRecallCurve,
)

from typing import Dict, Optional


class Metric:
    def __init__(
        self,
        average: str | None = "micro",
        num_classes: int = 1,
        device: Optional[torch.device] = None,
        output_path: str = "./out/",
    ):
        """
        Initialize the Metric class.
        This class initializes several metrics useful for evaluating the performance of a multiclass classification model.

        Args:
            average (Optional[str]): Type of averaging to perform on the data. Choices are 'micro', 'macro', and 'weighted'.
            num_classes (int): The number of classes in the classification task.
            device (Optional[torch.device]): The device on which the metric computations are performed.
            output_path (str): The path where the metric results will be saved.
        """
        self.metrics = {
            "Accuracy": MulticlassAccuracy(average=average, num_classes=num_classes, device=device),
            "Precision": MulticlassPrecision(average=average, num_classes=num_classes, device=device),
            "Recall": MulticlassRecall(average=average, num_classes=num_classes, device=device),
            "F1": MulticlassF1Score(average=average, num_classes=num_classes, device=device),
            "AUROC": MulticlassAUROC(average=average, num_classes=num_classes, device=device),
            "AUPRC": MulticlassAUPRC(average=average, num_classes=num_classes, device=device),
            "Accuracy_per_class": MulticlassAccuracy(average=None, num_classes=num_classes, device=device),
            "Precision_per_class": MulticlassPrecision(average=None, num_classes=num_classes, device=device),
            "Recall_per_class": MulticlassRecall(average=None, num_classes=num_classes, device=device),
            "F1_per_class": MulticlassF1Score(average=None, num_classes=num_classes, device=device),
            "AUROC_per_class": MulticlassAUROC(average=None, num_classes=num_classes, device=device),
            "AUPRC_per_class": MulticlassAUPRC(average=None, num_classes=num_classes, device=device),
            "ConfusionMatrix": MulticlassConfusionMatrix(num_classes=num_classes, device=device),
        }

        self.output_path = output_path
        self.metric_dict: Dict[str, torch.Tensor] = {}

    def update(self, out: torch.Tensor, labels: torch.Tensor):
        """
        Update the metrics based on the model outputs and true labels.

        Args:
            out (torch.Tensor): The output from the model (logits or probabilities).
            labels (torch.Tensor): The true labels for the input data.
        """
        for metric in self.metrics.values():
            metric.update(out, labels)

    def reset(self):
        """
        Reset all the metrics.
        """
        for metric in self.metrics.values():
            metric.reset()
        self.metric_dict.clear()

    def compute(self):
        """
        Compute the final metric values.
        """
        for name, metric in self.metrics.items():
            self.metric_dict[name] = metric.compute().cpu().tolist()

    def save_file(self):
        """
        Save the computed metrics to a file.
        """
        with open(self.output_path, "w") as f:
            json.dump(self.metric_dict, f, indent=2)
