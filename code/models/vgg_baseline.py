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
 
    def __init__(self, skip_connection = False):

        super(VGGDecoder, self).__init__()
        self.skip_connection = skip_connection
        skip= 3 if self.skip_connection else 2 
        
        if self.skip_connection:
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
            self.lstm_hidden_conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias = False)
            self.skip_conv0 = nn.Conv2d(in_channels=256*2, out_channels=256*2, kernel_size=1, padding=0, stride=1, bias = False)

            self.lstm_hidden_conv1 = nn.Conv2d(in_channels=128*2, out_channels=128*2, kernel_size=1, padding=0, stride=1, bias = False)
            self.skip_conv1 = nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=1, padding=0, stride=1, bias = False)
            
            self.lstm_hidden_conv2 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=1, padding=0, stride=1, bias = False)
            self.skip_conv2 = nn.Conv2d(in_channels=64*3, out_channels=64*3, kernel_size=1, padding=0, stride=1, bias = False)

        self.vgg_block_dec3 = nn.Sequential(
                    ConvBlock(256 * (skip-1), 256),
                    ConvBlock(256, 256),
                    ConvTransposeBlock(256,128)
                    )
        
        self.vgg_block_dec2 = nn.Sequential(
                    ConvBlock(128*skip, 128),
                    ConvTransposeBlock(128,64)
                    )
        
        self.vgg_block_dec1 = nn.Sequential(
                    ConvBlock(64*skip, 64),
                    nn.ConvTranspose2d(64, 1, 4, 2, 1),
                    nn.Sigmoid()
                    )
             
    def forward(self, x):
        
        if self.skip_connection:

            encoded, lstm_outputs = x

            lstm_skip_in0 = self.lrelu(self.lstm_hidden_conv0(lstm_outputs[2]))
            skip_in0 = self.lrelu(self.skip_conv0(torch.cat([encoded[2], lstm_skip_in0], 1)))
            v3 = self.vgg_block_dec3(skip_in0)

            lstm_skip_in1 = self.lrelu(self.lstm_hidden_conv1(torch.cat([v3, lstm_outputs[1]],1)))
            skip_in1 = self.lrelu(self.skip_conv1(torch.cat([encoded[1], lstm_skip_in1], 1)))
            v2 = self.vgg_block_dec2(skip_in1)

            lstm_skip_in2 = self.lrelu(self.lstm_hidden_conv2(torch.cat([v2, lstm_outputs[0]],1)))
            skip_in2 = self.lrelu(self.skip_conv2(torch.cat([encoded[0], lstm_skip_in2], 1)))
            v1 = self.vgg_block_dec1(skip_in2)
        
        else:
            lstm_outputs = x
            
            v3 = self.vgg_block_dec3(lstm_outputs[2])

            v2 = self.vgg_block_dec2(torch.cat([v3, lstm_outputs[1]], 1))

            v1 = self.vgg_block_dec1(torch.cat([v2, lstm_outputs[0]], 1))
            
        return v1
