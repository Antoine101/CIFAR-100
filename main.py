import os
from argparse import ArgumentParser
import warnings
import datamodule
import pipeline
import torch
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
    parser.add_argument("--accelerator", default=None) # Supported inputs: "gpu", "cpu"
    parser.add_argument("--devices", default=None) # Supported inputs: number of graphics cards or cpu cores used (integer starting from 1)
    args = parser.parse_args()

    # Print accelerator
    print(f"Accelerator: {args.accelerator}")

    # Print name of graphics card with index specified in arguments
    if args.accelerator == "gpu":
        if not args.devices:
            args.devices == "auto"
            n_gpus = torch.cuda.device_count()
            print(f"Using all {n_gpus} GPUs:")
            for i in range(n_gpus):
                print(f" - {torch.get_device_name(device=i)}")
        elif args.devices == "auto":
            n_gpus = torch.cuda.device_count()
            print(f"Using all {n_gpus} GPUs:")
            for i in range(n_gpus):
                print(f" - {torch.get_device_name(device=i)}")
        else:
            print(f"Using {args.devices} GPU(s):")
            for i in range(args.devices):
                print(f" - {torch.get_device_name(device=i)}")
    elif args.accelerator=="cpu":
        if not args.devices:
            args.devices = 1
        print(f"Cores used: {args.devices}")

    # Set number of workers (for dataloaders)
    num_workers = int(os.cpu_count() / 4)
    print(f"Number of workers used: {num_workers}")

    # Set maximum number of epochs to train for
    max_epochs = 10
    print(f"Maximum number of epochs: {max_epochs}")

    # Set the batch size
    batch_size = 256 if args.accelerator=="gpu" else 64
    print(f"Batch size: {batch_size}")

    # Set the initial learning rate
    learning_rate = 0.1
    print(f"Initial learning rate: {learning_rate}")    

    # Instantiate the datamodule
    dm = datamodule.CIFAR100DataModule(batch_size=batch_size, num_workers=num_workers)

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
                    devices=int(args.devices),
                    max_epochs=max_epochs, 
                    logger=tensorboard_logger,
                    log_every_n_steps = 1,
                    callbacks=[lr_monitor, early_stopping, checkpoint]
                    ) 

    # Instantiate the pipeline
    pipeline = pipeline.CIFAR100ResNet(learning_rate=learning_rate, batch_size=batch_size)  
    
    # Fit the trainer on the training set
    trainer.fit(pipeline, dm)

    # Test on the test set
    trainer.test(pipeline, dm)