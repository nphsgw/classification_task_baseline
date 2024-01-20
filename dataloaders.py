from torch.utils.data import Subset, DataLoader
from torchvision.datasets import DatasetFolder, VisionDataset


class CustomDatasetLoader:
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        """
        Initialize the CustomDatasetLoader class.

        Args:
            batch_size (int): The number of samples to load in each batch. Default is 32.
            num_workers (int): The number of subprocesses to use for data loading. Default is 1.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build_dataloaders(
        self,
        labeled_train_dataset: DatasetFolder | Subset | VisionDataset,
        labeled_val_dataset: DatasetFolder | Subset | VisionDataset | None,
        labeled_test_dataset: DatasetFolder | Subset | VisionDataset | None,
        unlabeled_test_dataset: DatasetFolder | Subset | VisionDataset | None,
    ):
        """
        Build DataLoader instances for labeled training, validation, testing datasets, and an unlabeled testing dataset.

        Args:
            labeled_train_dataset (DatasetFolder | Subset | VisionDataset): The labeled training dataset.
            labeled_val_dataset (Optional[DatasetFolder | Subset | VisionDataset]): The labeled validation dataset. Default is None.
            labeled_test_dataset (Optional[DatasetFolder | Subset | VisionDataset]): The labeled testing dataset. Default is None.
            unlabeled_test_dataset (Optional[DatasetFolder | Subset | VisionDataset]): The unlabeled testing dataset. Default is None.
        """
        self.labeled_train_loader = DataLoader(
            labeled_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        if labeled_val_dataset is not None:
            self.labeled_val_loader = DataLoader(
                labeled_val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
        if labeled_test_dataset is not None:
            self.labeled_test_loader = DataLoader(
                labeled_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
        if unlabeled_test_dataset is not None:
            self.unlabeled_test_loader = DataLoader(
                unlabeled_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
