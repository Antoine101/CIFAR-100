import os
import warnings
import datamodule
import pipeline
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*Your `val_dataloader` has `shuffle=True`.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*")


    gpus = min(1, torch.cuda.device_count())
    batch_size = 128 if gpus else 64
    num_workers = int(os.cpu_count() / 4)
    print(f"Number of workers used: {num_workers}")

    max_epochs = 200
    learning_rate = 0.1

    dm = datamodule.CIFAR100DataModule(batch_size=batch_size, num_workers=num_workers)

    # Instantiate the logger
    tensorboard_logger = TensorBoardLogger(save_dir="logs")

    # Instantiate early stopping based on epoch validation loss
    early_stopping = EarlyStopping("validation_loss", patience=20, verbose=True)

    # Instantiate a learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Instantiate a checkpoint callback
    checkpoint = ModelCheckpoint(
                            dirpath=f"./checkpoints/",
                            filename="{epoch}-{validation_loss:.2f}",
                            verbose=True,
                            monitor="validation_loss",
                            save_last = False,
                            save_top_k=1,      
                            mode="min",
                            save_weights_only=True
                            )

    trainer = Trainer(
                    gpus=gpus,
                    max_epochs=max_epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[lr_monitor, checkpoint]
                    ) 

    pipeline = pipeline.CIFAR100ResNet(learning_rate=learning_rate, batch_size=batch_size)  
    trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm)