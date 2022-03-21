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
 
    def __init__(self, skip_connection = False):

        super(DCGANDecoder, self).__init__()

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

        self.block3 = nn.Sequential(
                    ConvTransposeBlock(256 * (skip - 1), 256, 3, 1, 1),
                    ConvTransposeBlock(256, 128)
                    )
        
        self.block2 = nn.Sequential(
                    ConvTransposeBlock(128*skip, 128, 3, 1, 1),
                    ConvTransposeBlock(128, 64)
                    )

        self.block1 = nn.Sequential(
                    ConvTransposeBlock(64*skip, 64, 3, 1, 1),
                    nn.ConvTranspose2d(64, 1, 4, 2, 1),
                    nn.Sigmoid()
                    )
             
    def forward(self, x):

        if self.skip_connection:

            encoded, lstm_outputs = x

            lstm_skip_in0 = self.lrelu(self.lstm_hidden_conv0(lstm_outputs[2]))
            skip_in0 = self.lrelu(self.skip_conv0(torch.cat([encoded[2], lstm_skip_in0], 1)))
            b3 = self.vgg_block_dec3(skip_in0)

            lstm_skip_in1 = self.lrelu(self.lstm_hidden_conv1(torch.cat([b3, lstm_outputs[1]],1)))
            skip_in1 = self.lrelu(self.skip_conv1(torch.cat([encoded[1], lstm_skip_in1], 1)))
            b2 = self.vgg_block_dec2(skip_in1)

            lstm_skip_in2 = self.lrelu(self.lstm_hidden_conv2(torch.cat([b2, lstm_outputs[0]],1)))
            skip_in2 = self.lrelu(self.skip_conv2(torch.cat([encoded[0], lstm_skip_in2], 1)))
            b1 = self.vgg_block_dec1(skip_in2)

        else:
            _,lstm_outputs = x
            
            b3 = self.block3(lstm_outputs[2])

            b2 = self.block2(torch.cat([b3, lstm_outputs[1]], 1))

            b1 = self.block1(torch.cat([b2, lstm_outputs[0]], 1))

        return b1