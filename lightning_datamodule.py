import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from normstats import compute_normstats

class CIFAR100DataModule(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2):
        super().__init__()
        self.save_hyperparameters()
    

    def prepare_data(self):
        datasets.CIFAR100(root="./data", download=True, train=True)
        datasets.CIFAR100(root="./data", download=True, train=False)


    def setup(self, stage = None):

        train_set = datasets.CIFAR100(root="./data", train=True, transform=transforms.ToTensor())
        train_set_mean, train_set_std = compute_normstats(train_set)

        if stage == "fit" or stage is None:

            train_set_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=15),
                                            transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std, inplace=True)])
            validation_set_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std, inplace=True)])                                            

            # Get the train set images indices and shuffle them
            train_set_length = len(train_set)
            indices = list(range(train_set_length))
            np.random.seed(42)
            np.random.shuffle(indices)

            # Calculate the split point to have 10% of the train set as a validation set
            split = int(np.floor(0.9 * train_set_length))

            # Create a sampler for the train set (used in train_dataloader)
            self.train_sampler = SubsetRandomSampler(indices[:split])

            # Get the indices for the validation set (used in val_dataloader)
            self.validation_indices = indices[split:]

            # Create the train, validation and test sets
            self.cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=train_set_transforms)
            self.cifar100_validation = datasets.CIFAR100(root="./data", train=True, transform=validation_set_transforms)

            # Retrieve classes from the train set
            self.classes = self.cifar100_train.classes

        if stage == "test" or stage is None:

            test_set_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std, inplace=True)])

            self.cifar100_test = datasets.CIFAR100(root="./data", train=False, transform=test_set_transforms)


    def train_dataloader(self):
        cifar100_train = DataLoader(self.cifar100_train, batch_size=self.hparams.batch_size, sampler=self.train_sampler, num_workers=self.hparams.num_workers, pin_memory=True)
        return cifar100_train


    def val_dataloader(self):
        cifar100_validation = DataLoader(self.cifar100_validation, batch_size=self.hparams.batch_size, sampler=self.validation_indices, num_workers=self.hparams.num_workers, pin_memory=True)
        return cifar100_validation


    def test_dataloader(self):
        cifar100_test = DataLoader(self.cifar100_test, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
        return cifar100_test