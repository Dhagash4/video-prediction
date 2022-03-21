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


class DCGANEncoder(nn.Module):
 
    def __init__(self, in_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super(DCGANEncoder, self).__init__()

        self.in_size = in_size
        self.input_channels = in_size[0]
        self.kernels = kernels
        self.latent_dim = latent_dim

        """ Defining convolutional encoder """
        
        self.encoder = nn.ModuleList([ConvBlock(self.kernels[i], self.kernels[i+1]) for i in range(len(self.kernels)-1)])
        # self.encoder = nn.Sequential(*layers)
        if self.in_size[1] == 128:
            self.encoder.append(ConvBlock(self.kernels[-1], self.kernels[-1]))

        self.out_embed = nn.Sequential(nn.Conv2d(self.kernels[-1], latent_dim, 4, 1, 0),
                                       nn.BatchNorm2d(latent_dim),
                                       nn.Tanh())
         
    def forward(self,x):
        encoded_skip = []
        input = x

        for i, enc in enumerate(self.encoder):
            encoded_skip.append(enc(input.clone()))
            input = encoded_skip[i].clone()

        # for i in range(len(self.kernels)-1):
        #     encoded_skip.append(self.encoder[i](input.clone()))
        #     input = encoded_skip[i].clone()
        out = self.out_embed(input) 
        return out.view(-1, self.latent_dim), encoded_skip

class DCGANDecoder(nn.Module):
 
    def __init__(self, out_size = (1, 64, 64), kernels = [1, 64, 128, 256, 512], latent_dim=128):

        super(DCGANDecoder, self).__init__()

        self.out_size = out_size
        # self.input_channels = in_size[0]
        self.kernels = kernels[::-1]
        self.latent_dim = latent_dim

        """ The reason fro using  """
  
        self.c1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, self.kernels[0], 4, 1, 0),
                                       nn.BatchNorm2d(self.kernels[0]),
                                       nn.Tanh())

        self.decoder = nn.ModuleList([ConvTransposeBlock(self.kernels[i]*2, self.kernels[i+1]) for i in range(len(self.kernels)-2)])

        if self.out_size[1] == 128:
            self.decoder.insert(0, ConvTransposeBlock(self.kernels[0]*2, self.kernels[0]))

        self.out = nn.Sequential(nn.ConvTranspose2d(self.kernels[-2]*2, self.kernels[-1], 4, 2, 1),
                                       nn.Sigmoid())
             
    def forward(self, x):
        input, encoded_skip = x
        input = self.c1(input.clone().view(-1, self.latent_dim, 1, 1))
        n = len(encoded_skip)-1

        for i, dec in enumerate(self.decoder):
            input = dec(torch.cat([input.clone(), encoded_skip[n-i]],1))

        # for i in range(len(self.kernels)-2):
        #     input = self.dec[i](torch.cat([input.clone(), encoded_skip[3-i]],1))

        output = self.out(torch.cat([input, encoded_skip[0]],1))
        return output