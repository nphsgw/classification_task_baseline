import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from model_manager import ModelManager

# type hinting
from torch import Tensor
import torch.utils.data
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ImageClassificationTrainer:
    def __init__(
        self,
        model_manager: ModelManager,
        train_loader: DataLoader,
        val_loader: DataLoader,
        total_epochs: int = 1,
        log_dir: str = "./log/",
        checkpoint_dir: str = "./log/",
    ):
        """
        Initialize the ImageClassificationTrainer class.

        Args:
            model_manager (ModelManager): Manager handling the model operations.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            total_epochs (int): Total number of epochs for training.
            log_dir (str): Directory where TensorBoard logs will be saved.
            checkpoint_dir (str): Directory where model checkpoints will be saved.
        """
        self.model_manager = model_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = total_epochs
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir

    def train(self):
        for epoch in tqdm(range(self.total_epochs)):
            train_acc, train_loss = self.train_one_epoch(
                self.model_manager.model,
                self.model_manager.device,
                self.model_manager.loss_fn,
                self.model_manager.optimizer,
                self.model_manager.lr_scheduler,
                self.train_loader,
            )

            val_acc, val_loss = self.validate_one_epoch(
                self.model_manager.model,
                self.model_manager.device,
                self.model_manager.loss_fn,
                self.model_manager.optimizer,
                self.val_loader,
            )
            lr = self.model_manager.optimizer.param_groups[0]["lr"]
            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("lr", lr, epoch)

            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model_manager.model.state_dict(), self.checkpoint_dir + "model_" + str(epoch + 1) + ".pth"
                )

    def _process_batch(
        self,
        model: Module,
        device: torch.device,
        loss_fn: Module,
        optimizer: Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
        lr_scheduler: Optional[LRScheduler] = None,
        training: bool = True,
    ) -> Tuple[float, float]:
        """
        Process a single batch of data.

        Args:
            model (Module): The neural network model.
            device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
            loss_fn (Module): The loss function used to calculate the model's loss.
            optimizer (Optimizer): The optimizer used to update the model's weights.
            batch (Tuple[torch.Tensor, torch.Tensor]): Input data and labels.
            lr_scheduler (Optional[LRScheduler]): Learning rate scheduler to adjust the learning rate during training (default: None).
            training (bool): Flag indicating if the model is in training mode (default: True).

        Returns:
            Tuple[float, float]: Tuple containing the loss for the batch and the number of correct predictions.
        """
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)
        _, pred = torch.max(out, 1)

        if training:
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        return loss.item(), torch.sum(pred == y).item()

    def train_one_epoch(
        self,
        model: Module,
        device: torch.device,
        loss_fn: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Train the model for a single epoch.

        Args:
            model (Module): The neural network model to train.
            device (torch.device): The device on which the model and data are loaded (CPU or GPU).
            loss_fn (Module): The loss function used to calculate the model's loss.
            optimizer (Optimizer): The optimizer used to update the model's weights.
            lr_scheduler (LRScheduler): Learning rate scheduler to adjust the learning rate during training.
            train_loader (DataLoader): DataLoader for the training data, which provides batches of data.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and average accuracy for the epoch.
        """
        model.train()
        total_loss: Tensor = torch.zeros(1).to(device)
        total_acc: Tensor = torch.zeros(1).to(device)

        for batch in tqdm(train_loader, total=len(train_loader), leave=False, desc="[train]"):
            loss, correct = self._process_batch(model, device, loss_fn, optimizer, batch, lr_scheduler, training=True)
            total_loss += loss * len(batch[0])
            total_acc += correct

        avg_loss: Tensor = total_loss / len(train_loader.dataset)
        avg_acc: Tensor = total_acc / len(train_loader.dataset)
        return avg_acc.cpu().numpy(), avg_loss.cpu().numpy()

    def validate_one_epoch(
        self,
        model: Module,
        device: torch.device,
        loss_fn: Module,
        optimizer: Optimizer,
        val_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Validate the model for one epoch using the validation data.

        Args:
            model (Module): The neural network model.
            device (torch.device): The device on which the model and data are loaded (CPU or GPU).
            loss_fn (Module): The loss function used to calculate the model's loss.
            optimizer (Optimizer): The optimizer used for training (not used in validation, but kept for consistency with _process_batch).
            val_loader (DataLoader):DataLoader for the training data, which provides batches of data.

        Returns:
            Tuple[float, float]: Tuple containing the average accuracy and average loss for the validation set.
        """
        model.eval()
        total_loss: Tensor = torch.zeros(1).to(device)
        total_acc: Tensor = torch.zeros(1).to(device)

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), leave=False, desc="[val]"):
                loss, correct = self._process_batch(model, device, loss_fn, optimizer, batch, None, False)
                total_loss += loss * len(batch[0])
                total_acc += correct

        avg_loss: Tensor = total_loss / len(val_loader.dataset)
        avg_acc: Tensor = total_acc / len(val_loader.dataset)
        return avg_acc.cpu().numpy(), avg_loss.cpu().numpy()
