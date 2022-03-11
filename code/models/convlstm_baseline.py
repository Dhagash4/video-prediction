from turtle import forward
from models.convlstm import *
import torch
import torch.nn as nn
import torchvision



class LSTM_baseline(nn.Module):

    def __init__(self, input_dim = (62,32,32), hidden_dim = [64,128,256], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2 ,mode="zeros",  batch_size =40, bias=True, device = "cpu",downsample=2):

        assert mode in ["zeros", "random", "learned"]
        super(LSTM_baseline,self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim =  hidden_dim 
        self.kernels = kernels if self.num_layers>=2 else kernels[0]
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.return_all_layers = return_all_layers
        self.bias = bias
        self.downsample = downsample

        self.conv_lstms = []
        self.convlstm1 = predictor_lstm(input_dim = self.input_dim, 
                                        hidden_dim = [hidden_dim[0],hidden_dim[0]], 
                                        kernels = self.kernels, 
                                        return_all_layers = False,
                                        num_layers=self.num_layers, 
                                        mode=self.mode,  
                                        batch_size =self.batch_size, 
                                        bias=self.bias, 
                                        device = self.device)

        self.convlstm2 = predictor_lstm(input_dim = (self.input_dim[0] * self.downsample, self.input_dim[1] / self.downsample, self.input_dim[2] / self.downsample), 
                                        hidden_dim = [hidden_dim[1],hidden_dim[1]], 
                                        kernels = self.kernels, 
                                        return_all_layers = False,
                                        num_layers=self.num_layers, 
                                        mode=self.mode,  
                                        batch_size =self.batch_size, 
                                        bias=self.bias, 
                                        device = self.device)
        self.convlstm3 = predictor_lstm(input_dim = (self.input_dim[0] * self.downsample * 2, self.input_dim[1] / (self.downsample * 2), self.input_dim[2] / (self.downsample * 2)), 
                                        hidden_dim = [hidden_dim[2],hidden_dim[2]], 
                                        kernels = self.kernels, 
                                        return_all_layers = False,
                                        num_layers=self.num_layers, 
                                        mode=self.mode,  
                                        batch_size =self.batch_size, 
                                        bias=self.bias, 
                                        device = self.device)


    def forward(self,x):
        encodings = x 
        # out = []

        h1 = self.convlstm1(encodings[0])
        h2 = self.convlstm2(encodings[1])
        h3 = self.convlstm2(encodings[2])


        return [h1,h2,h3]

    def _init_state(self):
        



