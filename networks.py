import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.resnet import ResNet50_Weights

class JerseyNumberClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 100)

    def forward(self, input):
        return self.model_ft(input)

class JerseyNumberMulticlassClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])

        self.head1 = nn.Linear(512, 100)
        self.head2 = nn.Linear(512, 10)
        self.head3 = nn.Linear(512, 11)

    def forward(self, input):
        # get backbone features
        backbone_feats = self.backbone(input)

        backbone_feats =backbone_feats.reshape(backbone_feats.size(0), -1)

        # pass through heads
        h1 = self.head1(backbone_feats)
        h2 = self.head2(backbone_feats)
        h3 = self.head3(backbone_feats)
        return h1, h2, h3


class SimpleJerseyNumberClassifier(nn.Module):
    def __init__(self):
        super(SimpleJerseyNumberClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(65536, 2048)
        self.linear2 = nn.Linear(2048, 100)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x



# ResNet18 based model for binary classification
class LegibilityClassifier(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True
        self.model_ft.avgpool.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = sigmoid(x)
        return x

# ResNet34 based model for binary classification
class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = sigmoid(x)
        return x
    
class LegibilityClassifier50(nn.Module):
    def __init__(self, train=False, finetune=False):
        super().__init__()
        # Load the pre-trained ResNet-50 model
        self.model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers initially if finetune
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False 

        # Get the number of input features for the original fully connected layer
        num_ftrs = self.model_ft.fc.in_features

        # Replace the original fully connected layer with a new one for binary classification (1 output neuron)
        self.model_ft.fc = nn.Linear(num_ftrs, 1)

        # Ensure the new fully connected layer and the last convolutional block (layer4) are trainable
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        # Pass the input through the modified ResNet-50 model
        x = self.model_ft(x)
        # Apply the sigmoid activation function for binary classification
        x = sigmoid(x)
        return x

# ResNet18 based model for binary classification
class LegibilityClassifierTransformer(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.vit_b_16(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.heads.head.in_features
        self.model_ft.heads.head = nn.Linear(num_ftrs, 1)
        self.model_ft.heads.head.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = sigmoid(x)
        return x

# Classifier Model
class LegibilitySimpleClassifier(nn.Module):
    def __init__(self):
        super(LegibilitySimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(8192, 2048)
        self.linear2 = nn.Linear(2048, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.leaky_relu(self.linear1(x))
        x = sigmoid(self.linear2(x))
        return x
