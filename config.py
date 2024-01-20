import torch


class Config:
    def __init__(
        self,
        num_classes: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 64,
        num_workers: int = 4,
        device: torch.device = torch.device("cuda:0"),
        momentum: float = 0.0001,
        weight_decay: float = 0.0001,
        n_folds: int = 5,
        train_dir: str = "",
        test_dir: str = "",
        log_dir: str = "",
        checkpoint_dir: str = "",
        metric_file_path: str = "",
    ):
        """
        Initialize configuration for the training process.

        Args:
            num_classes (int): Number of classes for classification.
            train_ratio (float): Proportion of data used for training.
            test_ratio (float): Proportion of data used for testing.
            num_epochs (int): Number of epochs for training.
            lr (float): Learning rate.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
            device (torch.device): Device for training (e.g., 'cuda:0').
            momentum (float): Momentum for the optimizer.
            weight_decay (float): Weight decay (L2 penalty).
            n_folds (int): Number of folds for cross-validation.
            train_dir (str): Directory for training data.
            test_dir (str): Directory for test data.
            log_dir (str): Directory for saving logs.
            metric_file_path (str): Directory for saving metrics file.
        """

        self.num_classes = num_classes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_folds = n_folds
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.metric_file_path = metric_file_path
