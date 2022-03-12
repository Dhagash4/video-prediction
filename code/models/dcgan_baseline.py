from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, input_channels, output_channels, kernel_size=4, stride = 2, padding = 1):

        super(ConvBlock, self).__init__()
        # convolutional layer
        self.module = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return(self.module(x))


class ConvTransposeBlock(nn.Module):
    """
    Encapuslation of transposed convolutional block (conv + activation + pooling)
    """
    def __init__(self, input_channels, output_channels, kernel_size=4, stride = 2, padding=1):

        super(ConvTransposeBlock, self).__init__()
        # convolutional layer
        # nn.ConvTranspose2d()
        self.module = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x):
        return(self.module(x))


class DCGANEncoder(nn.Module):
 
    def __init__(self):

        super(DCGANEncoder, self).__init__()

        """ Defining convolutional encoder """

        self.block1 = nn.Sequential(
                    ConvBlock(1, 64),
                    ConvBlock(64, 64, 3, 1, 1)
                    )
        
        self.block2 = nn.Sequential(
                    ConvBlock(64, 128),
                    ConvBlock(128, 128, 3, 1, 1)
                    )

        self.block3 = nn.Sequential(
                    ConvBlock(128, 256),
                    ConvBlock(256, 256, 3, 1, 1)
                    )

    def forward(self,x):

        encoded_skips = []

        b1 = self.block1(x)
        encoded_skips.append(b1)

        b2 = self.block2(b1)
        encoded_skips.append(b2)

        b3 = self.block3(b2)
        encoded_skips.append(b3)
        
        return encoded_skips

class DCGANDecoder(nn.Module):
 
    def __init__(self):

        super(DCGANDecoder, self).__init__()

        self.block3 = nn.Sequential(
                    ConvTransposeBlock(256, 256, 3, 1, 1),
                    ConvTransposeBlock(256, 128)
                    )
        
        self.block2 = nn.Sequential(
                    ConvTransposeBlock(128*2, 128, 3, 1, 1),
                    ConvTransposeBlock(128, 64)
                    )

        self.block1 = nn.Sequential(
                    ConvTransposeBlock(64*2, 64, 3, 1, 1),
                    nn.ConvTranspose2d(64, 1, 4, 2, 1),
                    nn.Sigmoid()
                    )
             
    def forward(self, x):

        lstm_outputs = x
        
        b3 = self.block3(lstm_outputs[2])

        b2 = self.block2(torch.cat([b3, lstm_outputs[1]], 1))

        b1 = self.block1(torch.cat([b2, lstm_outputs[0]], 1))

        return b1