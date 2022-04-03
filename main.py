import os
import warnings

import datamodule
import pipeline

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":

    warnings.filterwarnings("ignore", ".*Deprecated in NumPy 1.20.*")

    gpus = min(1, torch.cuda.device_count())
    batch_size = 256 if gpus else 64
    num_workers = int(os.cpu_count() / 2)
    print(f"Number of workers used: {num_workers}")

    max_epochs = 100
    learning_rate = 2e-4

    dm = datamodule.CIFAR100DataModule(batch_size=batch_size, num_workers=num_workers)

    # Instantiate the logger
    tensorboard_logger = TensorBoardLogger(save_dir="logs")

    # Instantiate a learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
                    gpus=gpus,
                    max_epochs=max_epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[lr_monitor]
                    ) 

    pipeline = pipeline.CIFAR100ResNet(lr=learning_rate)  
    trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm)