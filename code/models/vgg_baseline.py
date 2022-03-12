# from base64 import encode
from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
import torchvision
from models.convlstm import *

class ConvBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, input_channels, output_channels, kernel_size=3, stride = 1, padding = 1):

        super(ConvBlock, self).__init__()
        # convolutional layer
        self.module = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return(self.module(x))


class ConvTransposeBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, input_channels, output_channels, kernel_size=4, stride = 2, padding=1):

        super(ConvTransposeBlock, self).__init__()
        # convolutional layer
        self.module = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x):
        return(self.module(x))


class VGGEncoder(nn.Module):
 
    def __init__(self):

        super(VGGEncoder, self).__init__()


        """ Defining convolutional encoder """
        self.vgg_block1 = nn.Sequential(
                    ConvBlock(1, 64),
                    ConvBlock(64, 64),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )
        
        self.vgg_block2 = nn.Sequential(
                    ConvBlock(64, 128),
                    ConvBlock(128, 128),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )
        
        self.vgg_block3 = nn.Sequential(
                    ConvBlock(128, 256),
                    ConvBlock(256, 256),
                    ConvBlock(256, 256),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )
         
    def forward(self,x):
        
        

        v1 = self.vgg_block1(x)
        

        v2 = self.vgg_block2(v1)
        

        v3 = self.vgg_block3(v2)
        
        
        return [v1,v2,v3]

class VGGDecoder(nn.Module):
 
    def __init__(self):

        super(VGGDecoder, self).__init__()

        
        
        self.vgg_block_dec3 = nn.Sequential(
                    ConvBlock(256, 256),
                    ConvBlock(256, 256),
                    ConvTransposeBlock(256,128)
                    )
        
        self.vgg_block_dec2 = nn.Sequential(
                    ConvBlock(128*2, 128),
                    ConvTransposeBlock(128,64)
                    )
        
        self.vgg_block_dec1 = nn.Sequential(
                    ConvBlock(64*2, 64),
                    nn.ConvTranspose2d(64, 1, 4, 2, 1),
                    nn.Sigmoid()
                    )
             
    def forward(self, x):
        
        lstm_outputs = x
        
        v3 = self.vgg_block_dec3(lstm_outputs[2])

        v2 = self.vgg_block_dec2(torch.cat([v3, lstm_outputs[1]], 1))

        v1 = self.vgg_block_dec1(torch.cat([v2, lstm_outputs[0]], 1))
       
        return v1
