from base64 import encode
from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
import torchvision

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
        # nn.ConvTranspose2d()
        self.module = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x):
        return(self.module(x))


class VGGEncoder(nn.Module):
 
    def __init__(self, in_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super(VGGEncoder, self).__init__()

        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim

        """ Defining convolutional encoder """
        self.vgg_block1 = nn.Sequential(
                    ConvBlock(self.input_channels, 64),
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

        self.vgg_block4 = None        
        if self.in_size[1] == 128:
            self.vgg_block4 = nn.Sequential(
                    ConvBlock(256, 512),
                    ConvBlock(512, 512),
                    ConvBlock(512, 512),
                    nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)    
                    )

            self.out_embed = nn.Sequential(nn.Conv2d(512, latent_dim, 3, 1, 1),
                                        nn.BatchNorm2d(latent_dim),
                                        nn.Tanh())

        else:
            self.out_embed = nn.Sequential(nn.Conv2d(256, latent_dim, 3, 1, 1),
                                        nn.BatchNorm2d(latent_dim),
                                        nn.Tanh())
         
    def forward(self,x):
        
        encoded_skip = []
        v1 = self.vgg_block1(x)
        encoded_skip.append(v1)

        v2 = self.vgg_block2(v1)
        encoded_skip.append(v2)

        v3 = self.vgg_block3(v2)
        encoded_skip.append(v3)
        
        if self.vgg_block4 is not None:
            v4 = self.vgg_block4(v3)
            encoded_skip.append(v4)

            out = self.out_embed(v4)
        else:
            out = self.out_embed(v3)
        
        # out = out.view(-1, 512, 4, 4)
        return out, encoded_skip

class VGGDecoder(nn.Module):
 
    def __init__(self, out_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super(VGGDecoder, self).__init__()

        self.out_size = out_size
        self.input_channels = out_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim

        

        """ Defining convolutional encoder """
        self.vgg_block_dec4 = None

        if self.out_size[1] == 128:
            self.in1 = nn.ConvTranspose2d(self.latent_dim, 512, 3, 1, 1)
            self.vgg_block_dec4 = nn.Sequential(
                    ConvBlock(512*2, 512),
                    ConvBlock(512, 512),
                    ConvTransposeBlock(512,256)
                    )
        else:
            self.in1 = nn.ConvTranspose2d(self.latent_dim, 256, 3, 1, 1)

        self.vgg_block_dec3 = nn.Sequential(
                    ConvBlock(256*2, 256),
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
        
        input, encoded_skip = x
        input = self.in1(input)

        if self.vgg_block_dec4 is not None:
            input = self.vgg_block_dec4(torch.cat([input, encoded_skip[3]], 1))
        
        v3 = self.vgg_block_dec3(torch.cat([input, encoded_skip[2]], 1))

        v2 = self.vgg_block_dec2(torch.cat([v3, encoded_skip[1]], 1))

        v1 = self.vgg_block_dec1(torch.cat([v2, encoded_skip[0]], 1))
       
        return v1
