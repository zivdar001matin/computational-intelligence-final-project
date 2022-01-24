import pytorch_lightning as pl

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder


class CarsDataModule(pl.LightningDataModule):
    def __init__(self,  data_dir:str = 'dataset/', batch_size:int = 32,
                        num_workers:int = 2, validation_ratio:int = 0.1,
                        test_ratio:int = 0.1, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.args = kwargs

        # transforms for images
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def prepare_data(self):
        # download dataset if it's needed!
        pass

    def setup(self):
        full_dataset = ImageFolder(self.data_dir, transform=self.transform)
        all_data_size = len(full_dataset)
        test_size       = int(all_data_size * self.test_ratio)
        validation_size = int(all_data_size * self.validation_ratio)
        train_size      = all_data_size - validation_size - test_size

        # split train set and validation set from original train set (not test set)
        self.train_dataset, self.validation_dataset, self.test_dataset = \
                random_split(full_dataset, [train_size, validation_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          self.batch_size,
                          shuffle=True,     # shuffle the dataset
                          num_workers=self.num_workers,    # load in 4 thread
                          pin_memory=True)  # pin memory in device (gpu)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                               self.batch_size,
                               num_workers=self.num_workers,
                               pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                               self.batch_size,
                               num_workers=self.num_workers,
                               pin_memory=True)

    def teardown(self):
        # Used to clean-up when the run is finished
        pass

