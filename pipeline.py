import model
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from model import create_model

class CIFAR100ResNet(LightningModule):
    def __init__(self, lr):
        super().__init__()
        
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters() 
        
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
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss    
        
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        
    def on_save_checkpoint(self, checkpoint):
        # Get the state_dict from self.model to get rid of the "model." prefix
        checkpoint["state_dict"] = self.state_dict()