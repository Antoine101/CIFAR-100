import os
from argparse import ArgumentParser
import utils
import warnings
import lightning_datamodule
import lightning_module
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":

    # Filter harmless warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*Your `val_dataloader` has `shuffle=True`.*")
    warnings.filterwarnings("ignore", ".*Checkpoint directory.*")

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu", help="Type of accelerator: 'gpu', 'cpu', 'auto'")
    parser.add_argument("--devices", default=1, help="Number of devices (GPUs or CPU cores) to use: integer starting from 1 or 'auto'")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to use as as workers for the dataloarders: integer starting from 1 to maximum number of cores on this machine")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of epochs to run for")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    args = parser.parse_args()

    # Print summary of selected arguments and adjust them if needed
    args = utils.args_interpreter(args)

    # Instantiate the datamodule
    dm = lightning_datamodule.CIFAR100DataModule(batch_size=args.bs, num_workers=args.workers)

    # Instantiate the logger
    tensorboard_logger = TensorBoardLogger(save_dir="logs")

    # Instantiate early stopping based on epoch validation loss
    early_stopping = EarlyStopping("validation_loss", patience=40, verbose=True)

    # Instantiate a learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

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

    # Instantiate the trainer
    trainer = Trainer(
                    accelerator=args.accelerator,
                    devices=args.devices,
                    max_epochs=args.epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[lr_monitor, early_stopping, checkpoint]
                    ) 

    # Instantiate the pipeline
    pipeline = lightning_module.CIFAR100ResNet(learning_rate=args.lr, batch_size=args.bs)  
    
    # Fit the trainer on the training set
    trainer.fit(pipeline, dm)

    # Test on the test set
    trainer.test(pipeline, dm)