import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ModelManager:
    def __init__(
        self,
        model: Module,
        device: torch.device,
        loss_fn: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ):
        """
        Initialize the ModelManager with the given model, device, loss function, optimizer, and optional learning rate scheduler.

        Args:
            model (Module): The neural network model to be trained.
            device (torch.device): The device (CPU or GPU) on which to run the model.
            loss_fn (Module): The loss function to be used for training.
            optimizer (Optimizer): The optimizer to be used for training.
            lr_scheduler (LRScheduler): Optional learning rate scheduler.
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._check_device_consistency()

    def update_model(self):
        pass

    def update_loss_fn(self):
        pass

    def update_optimizer(self):
        pass

    def update_lr_scheduler(self, len_train_loader, num_epochs):
        iterations = len_train_loader * num_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations)

    def _check_device_consistency(self):
        """
        Check if all components of the model are on the same device.
        Raises an error if any component is on a different device.
        """
        if hasattr(self.loss_fn, "weight") and self.loss_fn.weight is not None:
            if self.loss_fn.weight.device != self.device:
                raise ValueError(
                    f"Loss function's weight is on {self.loss_fn.weight.device}, but expected {self.device}"
                )
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                raise ValueError(f"Parameter {name} is on {param.device}, but expected {self.device}")
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.device != self.device:
                    raise ValueError(f"Optimizer parameter is on {param.device}, but expected {self.device}")
