import torch.nn as nn
import torchvision

def create_model():
    """
    Function to create the model.
    
    Takes in: -

    Returns: model
    """
    model = torchvision.models.resnet18(pretrained=True, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model