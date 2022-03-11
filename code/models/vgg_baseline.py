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
 
    def __init__(self, in_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128, device = "cpu"):

        super(VGGEncoder, self).__init__()

        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim
        self.device = device

        """ Defining convolutional encoder """
        self.vgg_block1 = nn.Sequential(
                    ConvBlock(self.input_channels, 64),
                    ConvBlock(64, 64),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )

        self.convlstm1 = predictor_lstm( input_dim = (64,32,32), hidden_dim = [64,64], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)
        
        self.vgg_block2 = nn.Sequential(
                    ConvBlock(64, 128),
                    ConvBlock(128, 128),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )

        self.convlstm2 = predictor_lstm( input_dim = (128,16,16), hidden_dim = [128,128], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)
        
        self.vgg_block3 = nn.Sequential(
                    ConvBlock(128, 256),
                    ConvBlock(256, 256),
                    ConvBlock(256, 256),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )

        self.convlstm3 = predictor_lstm( input_dim = (256,8,8), hidden_dim = [256,256], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)

         
    def forward(self,x):
        
        encoded_skip = []
        lstm_outputs =[]

        v1 = self.vgg_block1(x)
        lstm1 = self.convlstm1(v1)

        lstm_outputs.append(lstm1)
        encoded_skip.append(v1)

        v2 = self.vgg_block2(v1)
        lstm2 = self.convlstm2(v2)

        lstm_outputs.append(lstm2)
        encoded_skip.append(v2)

        v3 = self.vgg_block3(v2)
        lstm3 = self.convlstm1(v3)

        lstm_outputs.append(lstm3)
        encoded_skip.append(v3)
        
        return encoded_skip, lstm_outputs

class VGGDecoder(nn.Module):
 
    def __init__(self, out_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super(VGGDecoder, self).__init__()

        self.out_size = out_size
        self.input_channels = out_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim

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
                    nn.ConvTranspose2d(64, self.input_channels, 4, 2, 1),
                    nn.Sigmoid()
                    )
             
    def forward(self, x):
        
        encoded_skip, lstm_outputs = x
        # input = self.in1(input)

        # if self.vgg_block_dec4 is not None:
        #     input = self.vgg_block_dec4(torch.cat([input, encoded_skip[3]], 1))
        
        v3 = self.vgg_block_dec3(lstm_outputs[2])

        v2 = self.vgg_block_dec2(torch.cat([v3, lstm_outputs[1]], 1))

        v1 = self.vgg_block_dec1(torch.cat([v2, lstm_outputs[0]], 1))
       
        return v1