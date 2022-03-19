from unittest import skip
import torch
import torch.nn as nn

class ResidualBlockEncoder(nn.Module):
    
    def __init__(self, input_channels, output_channels, activation = "LeakyReLU", downsample = False):

        super().__init__()
        self.activation = activation

        if not downsample:
            output_channels = input_channels


        stride = 2 if downsample else 1  

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=stride, bias = False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2),
                                          nn.BatchNorm2d(output_channels))

    def forward(self,x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.lrelu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        out = y+x
        out = self.lrelu(out)
        return out
    
class Resnet18Encoder(nn.Module):
    
    def __init__(self):
 
        super().__init__()

        self.activation = "LeakyReLU"
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 64, kernel_size=3, padding=1, stride=1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, downsample=False)

        self.layer2 = self._make_layer(64, 128,  downsample=True)

        self.layer3 = self._make_layer(128, 256, downsample=True)

    def _make_layer(self, input_channels, output_channels, downsample=False):

        layers = []
        layers.append(ResidualBlockEncoder(input_channels, output_channels, activation = self.activation, downsample = downsample))

        if downsample:
            layers.append(ResidualBlockEncoder(output_channels, output_channels, activation = self.activation, downsample = False))

        else:
            layers.append(ResidualBlockEncoder(input_channels, input_channels, activation = self.activation, downsample = False))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        encoded_skips =[]

        input = self.c1(x)
        l1 = self.layer1(input)
        encoded_skips.append(l1)
        
        l2 = self.layer2(l1)
        encoded_skips.append(l2)

        l3 = self.layer3(l2)
        encoded_skips.append(l3)
        
        return encoded_skips

class ResidualBlockDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, activation = "LeakyReLU", upsample = False):

        super().__init__()
        self.activation = activation

        stride = 2 if upsample else 1  

        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1, stride=1, bias = False)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        if(stride==1):
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1, bias = False)
            self.bn1 = nn.BatchNorm2d(output_channels)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,  output_padding=1, stride=2, bias = False)
            self.bn1 = nn.BatchNorm2d(output_channels)

        self.upsample = None
        if upsample:
            # self.conv1 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,  output_padding=1, stride=2, bias = False)
            # self.bn1 = nn.BatchNorm2d(output_channels)
            self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,  output_padding=1, stride=2, bias = False),
                                          nn.BatchNorm2d(output_channels))


    def forward(self,x):


        y = self.conv2(x)
        y = self.bn2(y)
        y = self.lrelu(y)

        y = self.conv1(y)
        y = self.bn1(y)

        if self.upsample is not None:
            x = self.upsample(x)
        out = y+x
        out = self.lrelu(out)
        return out
    
class Resnet18Decoder(nn.Module):
    
    def __init__(self,skip_connection=False):
 
        super().__init__()
        self.activation = "LeakyReLU"
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        if not skip_connection:
            skip = 2
        else:
            skip = 2
            self.lstm_hidden_conv0 = nn.Conv2d(in_channels=256*2, out_channels=256*2, kernel_size=1, padding=0, stride=1, bias = False)
            self.lstm_hidden_conv1 = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=1, padding=0, stride=1, bias = False)
            self.skip_conv1 = nn.Conv2d(in_channels=128*2, out_channels=128*2, kernel_size=1, padding=0, stride=1, bias = False)
            self.lstm_hidden_conv2 = nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=1, padding=0, stride=1, bias = False)
            self.skip_conv2 = nn.Conv2d(in_channels=64*2, out_channels=64*2, kernel_size=1, padding=0, stride=1, bias = False)


        self.layer3 = self._make_layer(256*(skip-1), 128,  upsample=True)
        self.layer2 = self._make_layer(128*skip, 64, upsample=True)
        self.layer1 = self._make_layer(64*skip, 64, upsample=False)

        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)
        self.upc6 = nn.Sequential(
            nn.ConvTranspose2d(64*skip, 1, kernel_size=3, padding=1, stride=1),
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

    def forward(self,x,skip_connection=False):

        if not skip_connection:
  
            _,lstm_outputs = x

            l3 = self.layer3(lstm_outputs[2])

            l2 = self.layer2(torch.cat([l3, lstm_outputs[1]], 1))

            l1 = self.layer1(torch.cat([l2, lstm_outputs[0]], 1))
            
            out = self.upc6(self.upsamp(l1))
        
        else:

            encoded,lstm_outputs = x
            skip_in0 = self.lrelu(self.lstm_hidden_conv0(torch.cat([encoded[2], lstm_outputs[2]],1)))
            l3 = self.layer3(skip_in0)

            lstm_skip_in1 = self.lrelu(self.lstm_hidden_conv1(torch.cat([l3, lstm_outputs[1]],1)))
            skip_in1 = self.lrelu(self.skip_conv1(torch.cat([encoded[1], lstm_skip_in1], 1)))

            l2 = self.layer2(skip_in1)

            lstm_skip_in2 = self.lrelu(self.lstm_hidden_conv2(torch.cat([l2, lstm_outputs[0]],1)))
            skip_in2 = self.lrelu(self.skip_conv2(torch.cat([encoded[0], lstm_skip_in2], 1)))

            l1 = self.layer1(skip_in2)
            out = self.upc6(self.upsamp(l1))

        return out
