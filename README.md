# CIFAR-100
My attempt at CIFAR-100.

In this repository, you will find both a notebook version and a python script version of CIFAR-100.

The notebook version was created as a course project delivrable while the script version is to be able to increase num_workers for the dataloaders (Jupyter notebook is known to have issues handling multi-processing without additional libraries, so i decided to go for a .py version as well to speed up training).

Everything is done with pytorch-lightning and hpyer-parameters, model, image samples, histograms and metrics are logged to Tensorboard.

Have fun playing with it!
