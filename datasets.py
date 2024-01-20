import copy
import numpy as np

import torchvision
from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder, VisionDataset
from sklearn.model_selection import StratifiedShuffleSplit


class CustomDatasets:
    def __init__(
        self,
        standard_dataset_name: str = "",
        labeled_train_dataset_path: str = "./data/train",
        labeled_test_dataset_path: str = "",
        unlabeled_test_dataset_path: str = "",
        train_transform=None,
        test_transform=None,
    ):
        """
        Initialize the CustomDatasets class.

        Args:
            standard_dataset_name (str): Name of a standard dataset available in torchvision (if any).
            labeled_train_dataset_path (str): File path to the labeled training dataset.
            labeled_test_dataset_path (str): File path to the labeled testing dataset.
            unlabeled_test_dataset_path (str): File path to the unlabeled testing dataset.
            train_transform (Optional[torchvision.transforms.Compose]): Transformations to be applied to the training dataset.
            test_transform (Optional[torchvision.transforms.Compose]): Transformations to be applied to the testing dataset.
        """
        if not labeled_train_dataset_path:
            raise ValueError("dataset_path is empty. Please enter an appropriate string.")

        self.train_transform = train_transform
        self.test_transform = test_transform
        if standard_dataset_name == "":
            # Custom datasets
            self.labeled_train_dataset: DatasetFolder | VisionDataset = torchvision.datasets.ImageFolder(
                root=labeled_train_dataset_path, transform=self.train_transform
            )
            if labeled_test_dataset_path != "":
                self.labeled_test_dataset: DatasetFolder | VisionDataset = torchvision.datasets.ImageFolder(
                    root=labeled_test_dataset_path, transform=self.test_transform
                )
            if unlabeled_test_dataset_path != "":
                self.unlabeled_test_dataset: DatasetFolder | VisionDataset = torchvision.datasets.ImageFolder(
                    root=unlabeled_test_dataset_path, transform=self.test_transform
                )

        elif standard_dataset_name == "CIFAR10":
            # CIFAR10 datasets
            self.labeled_train_dataset = torchvision.datasets.CIFAR10(
                root="./standard_datasets/CIFAR10/train", train=True, download=True, transform=self.train_transform
            )

            self.labeled_test_dataset = torchvision.datasets.CIFAR10(
                root="./standard_datasets/CIFAR10/test", train=False, download=True, transform=self.test_transform
            )
        elif standard_dataset_name == "MNIST":
            self.labeled_train_dataset = torchvision.datasets.MNIST(
                root="./standard_datasets/MNIST/train", train=True, download=True, transform=self.train_transform
            )
            self.labeled_test_dataset = torchvision.datasets.MNIST(
                root="./standard_datasets/MNIST/test", train=False, download=True, transform=self.test_transform
            )
        else:
            raise ValueError("Unsupported dataset")

    def split_labeled_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        enable_test_split: bool = False,
        method: str = "StratifiedShuffleSplit",
    ):
        """
        Split the labeled dataset into training, validation, and optionally test datasets.

        Args:
            train_ratio (float): The proportion of the dataset to include in the train split.
            val_ratio (float): The proportion of the dataset to include in the validation split.
            enable_test_split (bool): Flag to determine if a separate test dataset should be created.
            method (str): The method to use for splitting the dataset. Currently supports 'StratifiedShuffleSplit'.

        Raises:
            ValueError: If the method is not 'StratifiedShuffleSplit' or etc.
        """
        labels = [label for _, label in self.labeled_train_dataset]
        if method == "StratifiedShuffleSplit":
            sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=0)
            train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
            # split train_dataset
            self.split_train_dataset = Subset(self.labeled_train_dataset, train_idx)
            if enable_test_split:
                test_rate = val_ratio / (1.0 - train_ratio)

                if test_rate > 0.5:
                    test_rate = 0.5

                sss_final = StratifiedShuffleSplit(n_splits=1, test_size=test_rate, random_state=0)
                temp_idx = copy.deepcopy(val_idx)
                val_idx, test_idx = next(sss_final.split(np.zeros(len(temp_idx)), [labels[i] for i in temp_idx]))

                val_idx = [temp_idx[i] for i in val_idx]
                test_idx = [temp_idx[i] for i in test_idx]

                if False:
                    # train, val, testデータセットに重複があるかを調べる
                    common_elements = set(train_idx).intersection(val_idx)
                    print(f"train_idxとval_idxの共通要素={common_elements}")
                    common_elements = set(val_idx).intersection(test_idx)
                    print(f"val_idxとtest_idxの共通要素={common_elements}")
                    common_elements = set(test_idx).intersection(train_idx)
                    print(f"test_idxとtrain_idxの共通要素={common_elements}")
                self.split_val_dataset = copy.deepcopy(Subset(self.labeled_train_dataset, val_idx))
                self.split_test_dataset = copy.deepcopy(Subset(self.labeled_train_dataset, test_idx))
                print("Split train_datasets => split_train_dataset, split_val_dataset and split_test_dataset")
                print(f"ratio train : val : test = {train_ratio} : {val_ratio} : {1.0 - train_ratio - val_ratio}")
                print(
                    f"train_dataset len={len(self.split_val_dataset)}, val_dataset len={len(self.split_val_dataset)}, test_dataset len={len(self.split_test_dataset)}"
                )
            else:
                self.split_val_dataset = copy.deepcopy(Subset(self.labeled_train_dataset, val_idx))
                print("Split train_datasets => split_train_dataset and split_val_dataset")
                print(f"ratio train : val = {train_ratio} : {val_ratio}")
        else:
            raise ValueError("Invalid method choice. Please select 'StratifiedShuffleSplit' or etc.")

    def set_transform(self, dataset, transform):
        if isinstance(dataset, Subset):
            self.set_transform(dataset.dataset, transform)
        else:
            dataset.transform = transform

    def print_set_transform(self, dataset, datasets_name):
        if isinstance(dataset, Subset):
            self.print_set_transform(dataset.dataset, datasets_name)
        else:
            print(f"{datasets_name} is set : {dataset.transform}")
