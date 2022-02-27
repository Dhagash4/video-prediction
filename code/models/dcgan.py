import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    """
    Encapuslation of a convolutional block (conv + activation + pooling)
    """
    def __init__(self, input_channels, output_channels, kernel_size=4, stride = 2, padding = 1):

        super().__init__()
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

        super().__init__()
        # convolutional layer
        # nn.ConvTranspose2d()
        self.module = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x):
        return(self.module(x))


class DCGANEncoder(nn.Module):
 
    def __init__(self, in_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super().__init__()

        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim

        """ Defining convolutional encoder """
        
        self.encoder = nn.ModuleList([ConvBlock(self.kernels[i], self.kernels[i+1]) for i in range(len(self.kernels)-1)])
        # self.encoder = nn.Sequential(*layers)
        self.out_embed = nn.Sequential(nn.Conv2d(self.kernels[-1], latent_dim, 4, 1, 0),
                                       nn.BatchNorm2d(latent_dim),
                                       nn.Tanh())
         
    def forward(self,x):
        encoded_skip = []
        input = x
        for i in range(len(self.kernels)-1):
            encoded_skip.append(self.encoder[i](input.clone()))
            input = encoded_skip[i].clone()
        out = self.out_embed(input) 
        return out.view(-1, self.latent_dim), encoded_skip

class DCGANDecoder(nn.Module):
 
    def __init__(self, kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super().__init__()

        # self.in_size = in_size
        # self.input_channels = in_size[0]
        self.kernels = kernels[::-1]
        self.latent_dim = latent_dim

        """ Defining convolutional encoder """
  
        self.c1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, self.kernels[0], 4, 1, 0),
                                       nn.BatchNorm2d(self.kernels[0]),
                                       nn.Tanh())
        self.dec = nn.ModuleList([ConvTransposeBlock(self.kernels[i]*2, self.kernels[i+1]) for i in range(len(self.kernels)-2)])
        self.out = nn.Sequential(nn.ConvTranspose2d(self.kernels[-2]*2, self.kernels[-1], 4, 2, 1),
                                       nn.Sigmoid())
             
    def forward(self, x):
        input, encoded_skip = x
        input = self.c1(input.clone().view(-1, self.latent_dim, 1, 1))
        for i in range(len(self.kernels)-2):
            input = self.dec[i](torch.cat([input.clone(), encoded_skip[3-i]],1))
        output = self.out(torch.cat([input, encoded_skip[0]],1))
        return output