import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metric

# type hint
from model_manager import ModelManager


class ImageClassifierInference:
    def __init__(
        self,
        model_manager: ModelManager,
        test_loader: DataLoader,
        num_classes: int,
        metric_file_path: str,
    ):
        """
        Initialize the ImageClassifierInference class.

        Args:
            model_manager (ModelManager): An instance of ModelManager for handling the model.
            test_loader (DataLoader): DataLoader for the test dataset.
            num_classes (int): Number of classes in the classification task.
            metric_file_path (str): File path to save the computed metrics.
        """
        self.model_manager = model_manager
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.val_metrics = metric.Metric(
            average="macro", num_classes=num_classes, device=torch.device("cpu"), output_path=metric_file_path
        )

    def predict(self, is_classes: bool = False):
        """
        Perform inference on the test dataset and return the predictions.

        Args:
            is_classes (bool): Flag to determine whether to return class predictions or probabilities.
        """
        if is_classes:
            self.model_manager.model.eval()
        pred_fun = torch.nn.Softmax(dim=1)
        preds = []
        for x, y_t in tqdm(self.test_loader):
            with torch.set_grad_enabled(False):
                x = x.to(self.model_manager.device)
                y = pred_fun(self.model_manager.model(x))
            if is_classes:
                self.val_metrics.update(y, y_t)
            y = y.cpu().numpy()
            y = [np.argmax(z) for z in y[:,]]
            preds.append(y)
        self.preds = np.concatenate(preds)
        if is_classes:
            self.val_metrics.compute()
            self.val_metrics.save_file()
            self.val_metrics.reset()

    def write_prediction(self):
        image_ids = [os.path.basename(path) for path, _ in self.test_loader.dataset.imgs]
        os.makedirs("./out", exist_ok=True)
        with open("./out/out.csv", "w") as f:
            f.write("id,label\n")
            for i, p in zip(image_ids, self.preds):
                f.write("{},{}\n".format(i, p))
