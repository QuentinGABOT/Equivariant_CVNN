import torch
import torch.nn as nn
import torchvision.models as models
import torchcvnn.nn.modules as c_nn

from .blocks import DoubleConv, DoubleLinear, SingleLinear, SingleConv

class MnistClassifier(nn.Module):
    """
    ResNet-like for MNIST (CVNN).
    Input: (B, 1, 28, 28).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(1, 32, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1),
            DoubleConv(32, 64, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            DoubleLinear(in_ch=64, out_ch=256, linear_mode='complex',
                         normalization='group', activation='CReLU'),
            c_nn.Dropout(0.5),
            DoubleLinear(in_ch=256, out_ch=num_classes, linear_mode='complex',
                         normalization='group', activation='CReLU'),
        )
        self.avg_pooling = c_nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, x.abs()
    
    def get_features(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x

class Cifar10Classifier(nn.Module):
    """
    ResNet-like for CIFAR-10 (CVNN).
    Input: (B, 3, 32, 32).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(3, 64, conv_mode='complex', normalization="group", activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            c_nn.Dropout2d(0.1),
            
            DoubleConv(64, 128, conv_mode='complex', normalization="group", activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            c_nn.Dropout2d(0.1),
    
            DoubleConv(128, 256, conv_mode='complex', normalization="group", activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            )
        self.classifier = nn.Sequential(
            DoubleLinear(in_ch=256, out_ch=512, linear_mode='complex',
                         normalization="group", activation="CReLU"),
            c_nn.Dropout(0.5),
            DoubleLinear(in_ch=512, out_ch=num_classes, linear_mode='complex',
                         normalization="group", activation="CReLU"),        
            )
        self.avg_pooling = c_nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, x.abs()
    
    def get_features(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x
    
class MstarClassifier(nn.Module):
    """
    ResNet-like for MSTAR (CVNN).
    Input: (B, 1, 32, 32).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(1, 32, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(32, 64, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(64, 128, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(128, 256, conv_mode='complex', normalization='group', activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),

        )
        self.classifier = nn.Sequential(
            DoubleLinear(in_ch=256, out_ch=512, linear_mode='complex',
                         normalization='group', activation='CReLU'),
            DoubleLinear(in_ch=512, out_ch=num_classes, linear_mode='complex',
                         normalization='group', activation='CReLU'),    
        )
        self.avg_pooling = c_nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, x.abs()
    
    def get_features(self, x):
        """Extract the penultimate layer for FID."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Replay the linear part up to the penultimate layer
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x
    
class MRIReconstructor(nn.Module):
    """
    ResNet-like for MRI reconstruction (CVNN).
    Input: (B, 1, 32, 32).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            DoubleConv(1, 64, conv_mode='complex', normalization=None, activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
            DoubleConv(64, 128, conv_mode='complex', normalization=None, activation='CReLU',
                       kernel_size=3, stride=2, padding=1, residual=True),
        )
        self.bottleneck = nn.Sequential(
            SingleLinear(in_ch=128*8*8, out_ch=128, linear_mode='complex',
                         normalization=None, activation=None),
            SingleLinear(in_ch=128, out_ch=128*8*8, linear_mode='complex',
                         normalization=None, activation=None),
        )
        self.reconstructor = nn.Sequential(
            c_nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            DoubleConv(64, 64, conv_mode='complex', normalization=None, activation='CReLU',
                       kernel_size=3, stride=1, padding=1, residual=True),
            c_nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            SingleConv(32, 1, conv_mode='complex', normalization=None, activation=None,
                       kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.reconstructor(x)
        return None, x

    def get_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        for layer in self.bottleneck[:-1]:
            x = layer(x)
        return x