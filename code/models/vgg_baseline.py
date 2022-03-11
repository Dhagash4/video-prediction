from models.vgg import *
import torch
import torch.nn as nn
import torchvision



class VGG_Encoder(nn.Module):
    def __init__(self, in_size = (1, 64, 64), kernels = [1, 64, 128, 256]):
    
        super(VGG_Encoder, self).__init__()

        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels
        # self.latent_dim = latent_dim
        
        
        
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

        

    def forward(self,x):

        encoded_skip = []

        v1 = self.vgg_block1(x)
        encoded_skip.append(v1)

        v2 = self.vgg_block2(v1)
        encoded_skip.append(v2)

        out = self.vgg_block3(v2)
        encoded_skip.append(out)

        return out, encoded_skip


class VGG_Decoder(nn.Module):

    def __init__(self, out_size = (1, 64, 64), kernels = [1, 64, 128, 256]):
    
        super(VGG_Decoder, self).__init__()

        self.out_size = out_size
        self.out_channels = out_size[0]
        self.kernels = kernels        
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
                    nn.ConvTranspose2d(64, self.out_channels, 4, 2, 1),
                    nn.Sigmoid()
                    )

        

    def forward(self, x):
        
        input, encoded_skip = x

        
        v3 = self.vgg_block_dec3(torch.cat([input, encoded_skip[2]], 1))

        v2 = self.vgg_block_dec2(torch.cat([v3, encoded_skip[1]], 1))

        v1 = self.vgg_block_dec1(torch.cat([v2, encoded_skip[0]], 1))
       
        return v1

