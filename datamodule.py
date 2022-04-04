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
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        datasets.CIFAR100(root="./data", download=True, train=True)
        datasets.CIFAR100(root="./data", download=True, train=False)

    def setup(self, stage=None):
        train_set = datasets.CIFAR100(root="./data", train=True, transform=transforms.ToTensor())
        train_set_mean, train_set_std = compute_normstats(train_set)
        train_set_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std)])

        validation_set_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std)])                                            

        test_set_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std)])

        
        train_set_length = len(train_set)
        indices = list(range(train_set_length))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(np.floor(0.9 * train_set_length))

        self.train_sampler = SubsetRandomSampler(indices[:split])
        self.validation_indices = indices[split:]

        self.cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=train_set_transforms)
        self.cifar100_validation = datasets.CIFAR100(root="./data", train=True, transform=validation_set_transforms)
        self.cifar100_test = datasets.CIFAR100(root="./data", train=False, transform=test_set_transforms)

    def train_dataloader(self):
        cifar100_train = DataLoader(self.cifar100_train, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)
        return cifar100_train

    def val_dataloader(self):
        cifar100_validation = DataLoader(self.cifar100_validation, batch_size=self.batch_size, sampler=self.validation_indices, num_workers=self.num_workers)
        return cifar100_validation

    def test_dataloader(self):
        cifar100_test = DataLoader(self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return cifar100_test