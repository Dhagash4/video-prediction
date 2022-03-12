from models.convlstm import *
import torch
import torch.nn as nn
import torchvision

class predictor(nn.Module):
 
    def __init__(self, device = "cpu"):

        super(predictor, self).__init__()

        self.device = device

        self.convlstm1 = predictor_lstm(input_dim = (64,32,32), hidden_dim = 64, kernel_sizes = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)
        

        self.convlstm2 = predictor_lstm(input_dim = (128,16,16), hidden_dim =128, kernel_sizes = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)
        

        self.convlstm3 = predictor_lstm(input_dim = (256,8,8), hidden_dim = 256, kernel_sizes = [(5,5),(3,3)], return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size =40, bias=True, device = self.device)

         
    def forward(self,x):
        
        encoded_skips = x
        lstm_outputs =[]

        lstm1 = self.convlstm1(encoded_skips[0])
        lstm_outputs.append(lstm1)

        lstm2 = self.convlstm2(encoded_skips[1])
        lstm_outputs.append(lstm2)

        lstm3 = self.convlstm3(encoded_skips[2])
        lstm_outputs.append(lstm3)
        
        return lstm_outputs


    def init_hidden_states(self):
        self.convlstm1.hidden_state = self.convlstm1._init_hidden()
        self.convlstm2.hidden_state = self.convlstm2._init_hidden()
        self.convlstm3.hidden_state = self.convlstm3._init_hidden()

