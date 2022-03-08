import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image


def get_act(act_name, inplace = True):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh" ]
    activation = getattr(nn, act_name)
    return activation(inplace = True)


class ResidualBlockEncoder(nn.Module):
    
    def __init__(self, input_channels, output_channels, activation = "LeakyReLU", downsample = False):

        super().__init__()
        self.activation = activation

        if not downsample:
            output_channels = input_channels


        stride = 2 if downsample else 1  

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=stride, bias = False)
        self.bn1 = nn.BatchNorm2d(output_channels)
#         self.relu = get_act(self.activation, inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2),
                                          nn.BatchNorm2d(output_channels))

        # self.act = self.activation
    
    def forward(self,x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        out = y+x
#         out = get_act(self.activation, inplace=True)(out)
        out = self.relu(out)
        return out
    
class Resnet18Encoder(nn.Module):
    
    def __init__(self, in_size = (1,64,64), kernels = [64, 128, 256, 512], latent_dim = 128, activation = "LeakyReLU"):
 
        super().__init__()

        self.activation = activation
        # self.activation =  get_act(activation, inplace=True)
        self.latent_dim = latent_dim
        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.kernels[0], kernel_size=3, padding=1, stride=1, bias = False),
            nn.BatchNorm2d(self.kernels[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(self.kernels[0], self.kernels[0],  downsample=False)
        self.layer2 = self._make_layer(self.kernels[0], self.kernels[1],  downsample=True)
        self.layer3 = self._make_layer(self.kernels[1], self.kernels[2], downsample=True)
        self.layer4 = self._make_layer(self.kernels[2], self.kernels[3], downsample=True)
        
        self.layer5 = None
        if  self.in_size[1] == 128:
            self.layer5 = self._make_layer(self.kernels[3], self.kernels[3], downsample=True)

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.latent_embed = nn.Linear(512, 2*self.latent_dim )

        self.c6 = nn.Sequential(
                nn.Conv2d(512, self.latent_dim, 4, 1, 0),
                nn.BatchNorm2d(self.latent_dim),
                nn.Tanh()
        )

    def _make_layer(self, input_channels, output_channels, downsample=False):

        layers = []
        layers.append(ResidualBlockEncoder(input_channels, output_channels, activation = self.activation, downsample = downsample))

        if downsample:
            layers.append(ResidualBlockEncoder(output_channels, output_channels, activation = self.activation, downsample = False))

        else:
            layers.append(ResidualBlockEncoder(input_channels, input_channels, activation = self.activation, downsample = False))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        #output list for skip connection
        encoded_skip =[]

        input = self.c1(x)
        encoded_skip.append(input)

        l1 = self.layer1(input)
        encoded_skip.append(l1)
        
        l2 = self.layer2(l1)
        encoded_skip.append(l2)

        l3 = self.layer3(l2)
        encoded_skip.append(l3)

        l4 = self.layer4(l3)
        encoded_skip.append(l4)

        if self.layer5 is not None:
            l5 = self.layer5(l4)
            encoded_skip.append(l5)
            y = self.c6(l5)
        else:
            y = self.c6(l4)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.latent_embed(x)

        # mu = x[:, :self.latent_dim]
        # logvar = x[:, self.latent_dim:]
        
        return y.view(-1,self.latent_dim), encoded_skip

class ResidualBlockDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, activation = "ReLU", upsample = False):

        super().__init__()
        self.activation = activation

        # if not upsample:
        #   output_channels = input_channels


        stride = 2 if upsample else 1  

        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1, bias = False)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if(stride==1):
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1, bias = False)
            self.bn1 = nn.BatchNorm2d(output_channels)

        self.upsample = None
        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,  output_padding=1, stride=2, bias = False)
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,  output_padding=1, stride=2, bias = False),
                                          nn.BatchNorm2d(output_channels))

        # self.act = self.activation

    def forward(self,x):

        y = self.conv2(x)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv1(y)
        y = self.bn1(y)

        if self.upsample is not None:
            x = self.upsample(x)

        out = y+x
        out = self.relu(out)
        return out
    
class Resnet18Decoder(nn.Module):
    
    def __init__(self, out_size = (1,64,64), kernels = [64, 128, 256, 512], latent_dim = 128, activation = "ReLU"):
 
        super().__init__()

        self.activation = activation
        # self.activation =  get_act(activation, inplace=True)
        self.latent_dim = latent_dim
        self.input_channels = out_size[0]
        self.out_size = out_size 
        self.kernels = kernels

        # self.inv_latent_embed = nn.Linear(self.latent_dim, 512)

        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.layer5 = None
        if  self.out_size[1] == 128:
            self.layer5 = self._make_layer(2*self.kernels[-1], self.kernels[-1], upsample=True)

        self.layer4 = self._make_layer(2*self.kernels[-1], self.kernels[-2],  upsample=True)
        self.layer3 = self._make_layer(2*self.kernels[-2], self.kernels[-3],  upsample=True)
        self.layer2 = self._make_layer(2*self.kernels[-3], self.kernels[-4], upsample=True)
        self.layer1 = self._make_layer(2*self.kernels[-4], self.kernels[-4], upsample=True)
        
        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)
        self.upc6 = nn.Sequential(
            nn.ConvTranspose2d(2*64, self.input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
            )

    def _make_layer(self, input_channels, output_channels, upsample=False):

        layers = []
        layers.append(ResidualBlockDecoder(input_channels, input_channels, activation = self.activation, upsample = False))

        if upsample:
            layers.append(ResidualBlockDecoder(input_channels, output_channels, activation = self.activation, upsample = True))

        else:
            layers.append(ResidualBlockDecoder(input_channels, input_channels, activation = self.activation, upsample = False))

        return nn.Sequential(*layers)

    def forward(self,x):

        # x = self.inv_latent_embed(x)
        # x = x.view(x.size(0), 512, 1, 1)
        # x = F.interpolate(x, scale_factor=4)
        input, encoded_skip = x

        up1 = self.upc1(input.view(-1, self.latent_dim, 1, 1))
#         print(f"up1:{up1.shape}")

        if self.layer5 is not None:
#             print("encoded_skip 5:", encoded_skip[5].shape)
            up1 = self.layer5(torch.cat([up1, encoded_skip[5]], 1))
            # encoded_skip.append(x)
#         print(f"-up1:{up1.shape}")
        l4 = self.layer4(torch.cat([up1, encoded_skip[4]], 1))
#         print("encoded_skip 4:", encoded_skip[4].shape)
#         print("l4:", l4.shape)
        
        l3 = self.layer3(torch.cat([l4, encoded_skip[3]], 1))
#         print("encoded_skip 3:", encoded_skip[3].shape)
#         print("l3:", l3.shape)
        
        l2 = self.layer2(torch.cat([l3, encoded_skip[2]], 1))
#         print("encoded_skip 2:", encoded_skip[2].shape)
#         print("l2:", l2.shape)
        
        l1 = self.layer1(torch.cat([l2, encoded_skip[1]], 1))
#         print("encoded_skip 1:", encoded_skip[1].shape)
#         print("l1:", l1.shape)

        # x = torch.sigmoid(self.conv1(x))
        # x = F.interpolate(x, scale_factor=2)
#         print("encoded_skip 0:", encoded_skip[0].shape)
        out = self.upc6(torch.cat([l1, self.upsamp(encoded_skip[0])], 1))
#         print("encoded_skip 0:", encoded_skip[0].shape)
#         print("out:", out.shape)
#         print("l1:", l1.shape)
        # x = x.view(-1, 1, 128, 128)
        return out