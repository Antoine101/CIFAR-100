import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from model import create_model
 
class CIFAR100ResNet(LightningModule):
    def __init__(self, lr):
        super().__init__()
        
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters() 

        self.n_classes = 100
        self.confmat = ConfusionMatrix(num_classes=100)  
        
        # Creation of the model
        self.model = create_model()
   
        # Instatiation of the learning rate
        self.lr = lr
        
    def configure_optimizers(self):  
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 5000
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"inputs":x, "targets":y, "predictions":preds, "loss":loss}    

    def training_epoch_end(self, outputs):
        # Log weights and biases for all layers of the model
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params,self.current_epoch)
        # Only after the first training epoch, log one of the training inputs as a figure and log the model graph
        if self.current_epoch == 0:
            input_sample = outputs[0]["inputs"][0]
            input_sample = torch.unsqueeze(input_sample, 3)
            input_sample = torch.permute(input_sample, (0,3,1,2))
            self.logger.log_graph(self, input_sample)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log(f"validation_loss", loss, prog_bar=True)
        self.log(f"validation_acc", acc, prog_bar=True)
        return {"inputs":x, "targets":y, "predictions":preds, "loss":loss} 

    def validation_epoch_end(self, outputs):
        # Concatenate the targets of all batches
        targets = torch.cat([output["targets"] for output in outputs])
        # Concatenate the predictions of all batches
        preds = torch.cat([output["predictions"] for output in outputs])
        # Compute the confusion matrix, turn it into a DataFrame, generate the plot and log it
        cm = self.confmat(preds, targets)
        cm = cm.cpu()
        for class_id in range(self.n_classes):
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                precision = round(precision.item()*100,1)
                self.log(f"validation_precision/{self.n_classes}", precision)
        df_cm = pd.DataFrame(cm.numpy(), index = range(self.n_classes), columns=range(self.n_classes))
        plt.figure()
        fig = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.yticks(rotation=0)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_acc", acc, prog_bar=True)
        return {"targets":y, "predictions":preds}

    def test_epoch_end(self, outputs):
        targets = torch.cat([output["targets"] for output in outputs])
        preds = torch.cat([output["predictions"] for output in outputs])
        acc = accuracy(preds, targets)
        cm = self.confmat(preds, targets)
        cm = cm.cpu()
        with open("test_set_predictions.txt", "w") as f:
            f.write("==================================================\n")
            f.write("ACCURACY\n")
            f.write("==================================================\n")
            f.write("\n")            
            f.write(f"Total accuracy: {round(acc.item()*100, 1)}%\n")
            f.write("\n")
            f.write("Class ID - Acurracy (%)\n")
            for class_id in range(self.n_classes):
                precision = cm[class_id, class_id] / torch.sum(cm[:,class_id])
                f.write(f"Number of good predictions: {cm[class_id, class_id]}\n")
                f.write(f"Number of images of this class to predict: {torch.sum(cm[class_id, :])}\n")                
                precision = round(precision.item()*100, 1)
                f.write(f"{class_id} - {precision}\n")
            f.write("\n")
            f.write("\n")
            f.write("==================================================\n")
            f.write("PREDICTIONS DETAIL\n")
            f.write("==================================================\n")
            f.write("Image index - Target class ID - Predicted class ID\n")
            for i in range(len(targets)):
                f.write(f"{i} - {targets[i]} - {preds[i]}\n")
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()