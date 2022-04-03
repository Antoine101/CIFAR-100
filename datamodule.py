from torch.utils.data import random_split, DataLoader
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
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std)])

        test_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(train_set_mean, train_set_std)])

        self.cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=train_transforms)
        self.cifar100_test = datasets.CIFAR100(root="./data", train=False, transform=test_transforms)

    def train_dataloader(self):
        cifar100_train = DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return cifar100_train

    def test_dataloader(self):
        cifar100_test = DataLoader(self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return cifar100_test