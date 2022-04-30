import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, MultiStepLR
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from model import create_model
from torchvision.utils import make_grid
 
class CIFAR100ResNet(LightningModule):

    def __init__(self, learning_rate, batch_size):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()   
        # Creation of the model
        self.model = create_model()
        # Instantiation of metrics
        self.confmat = ConfusionMatrix(num_classes=100)    
        # Instantiation of the number of classes
        self.n_classes = 100 
        # Instantiation of the learning rate
        self.learning_rate = learning_rate
        # Instantiation of the batch_size
        self.batch_size = batch_size


    def configure_optimizers(self): 
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        #scheduler_dict = {
        #    "scheduler": MultiStepLR(
        #        optimizer,
        #        milestones=[60,120,160],
        #        gamma=0.2
        #        ),
        #    "interval": "epoch"
        #}      
        scheduler_dict = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=20
                ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "validation_loss"
        }  
        #steps_per_epoch = int(np.ceil(45000 / self.batch_size))
        #scheduler_dict = {
            #"scheduler": OneCycleLR(
            #    optimizer,
            #    max_lr=0.1,
            #    epochs=self.trainer.max_epochs,
            #    steps_per_epoch=steps_per_epoch
            #),
            #"interval": "step"
        #}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

   
    def forward(self, x):
        logits = self.model(x)
        return logits


    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"inputs":inputs, "targets":targets, "predictions":predictions, "loss":loss}    


    def training_epoch_end(self, outputs):
        # Log weights and biases for all layers of the model
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)
        # Only after the first training epoch, log one of the training inputs as a figure and log the model graph
        if self.current_epoch == 0:
            image_samples = outputs[0]["inputs"][:10]
            image_samples = image_samples.cpu()
            image_samples_grid = make_grid(image_samples, normalize=True)
            image_samples_grid = image_samples_grid.numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(np.transpose(image_samples_grid, (1, 2, 0)))
            self.logger.experiment.add_figure(f"Training sample normalized images", fig)
            input_sample = outputs[0]["inputs"][0]
            input_sample = torch.unsqueeze(input_sample, 3)
            input_sample = torch.permute(input_sample, (3,0,1,2))
            self.logger.experiment.add_graph(self, input_sample)


    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        acc = accuracy(predictions, targets)
        self.log(f"validation_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"validation_acc", acc, on_epoch=True, prog_bar=True)
        return {"inputs":inputs, "targets":targets, "predictions":predictions, "loss":loss} 


    def validation_epoch_end(self, outputs):
        # Concatenate the targets of all batches
        targets = torch.cat([output["targets"] for output in outputs])
        # Concatenate the predictions of all batches
        predictions = torch.cat([output["predictions"] for output in outputs])
        # Compute the confusion matrix
        cm = self.confmat(predictions, targets)
        # Send it to the CPU
        cm = cm.cpu()
        # For each class
        for class_id in range(self.n_classes):
                # Calculate and log its prediction precision on the full validation set
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                precision = round(precision.item()*100,1)
                self.log(f"validation_precision/{self.n_classes}", precision)


    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        acc = accuracy(predictions, targets)
        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_acc", acc, prog_bar=True)
        return {"targets":targets, "predictions":predictions, "probabilities":probabilities}


    def test_epoch_end(self, outputs):
        targets = torch.cat([output["targets"] for output in outputs])
        predictions = torch.cat([output["predictions"] for output in outputs])
        probabilities = torch.cat([output["probabilities"] for output in outputs])
        # Compute the total prediction accuracy on the full test set
        acc = accuracy(predictions, targets)
        # Compute the confusion matrix
        cm = self.confmat(predictions, targets)
        # Send it to the CPU
        cm = cm.cpu()
        # Calculate the accuracy for each class
        classes_precisions = []
        for class_id in range(self.n_classes):
            precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])            
            precision = round(precision.item()*100, 1)
            classes_precisions.append(precision)
        # Write the test set prediction performances to an csv file
        with open("test_set_predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.trainer.datamodule.classes)
            for _, image_probs in enumerate(probabilities.cpu().numpy()):
                writer.writerow(np.around(image_probs, decimals=2))
        # Write the test set prediction performances to a text file
        with open("test_set_predictions.txt", "w") as f:
            f.write("==================================================\n")
            f.write("ACCURACY\n")
            f.write("==================================================\n")
            f.write("\n")            
            f.write(f"Total: {round(acc.item()*100, 1)}%\n")
            f.write("\n")
            f.write("Per Class:\n")
            f.write("Class - Accuracy (%)\n")
            for class_id in range(self.n_classes):
                f.write(f"{self.trainer.datamodule.classes[class_id]} - {classes_precisions[class_id]}\n")
            f.write("\n")
            f.write("\n")
            f.write("==================================================\n")
            f.write("PREDICTIONS DETAIL\n")
            f.write("==================================================\n")
            f.write("Image index - Target class - Predicted class\n")
            for i in range(len(targets)):
                f.write(f"{i} - {self.trainer.datamodule.classes[targets[i]]} - {self.trainer.datamodule.classes[predictions[i]]}\n")
        

    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()