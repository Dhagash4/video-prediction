from models.convlstm import *
import torch
import torch.nn as nn
import torchvision

class predictor(nn.Module):
 
    def __init__(self, batch_size= 20, device = "cpu",mode="zeros",num_layers=2):

        super(predictor, self).__init__()
        assert mode in ["zeros", "random", "learned"]
        self.device = device
        self.mode = mode
        

        self.convlstm1 = predictor_lstm( input_dim = (64,32,32), hidden_dim = [64,64], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=num_layers, mode=self.mode,  batch_size =batch_size, bias=True, device = self.device)
        

        self.convlstm2 = predictor_lstm( input_dim = (128,16,16), hidden_dim = [128,128], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=num_layers, mode=self.mode,  batch_size =batch_size, bias=True, device = self.device)
        

        self.convlstm3 = predictor_lstm( input_dim = (256,8,8), hidden_dim = [256,256], kernels = [(5,5),(3,3)], return_all_layers = False,
                num_layers=num_layers, mode=self.mode,  batch_size =batch_size, bias=True, device = self.device)

         
    def forward(self,x):
        
        encoded_skips = x

        lstm1 = self.convlstm1(encoded_skips[0])
        

        lstm2 = self.convlstm2(encoded_skips[1])
        

        lstm3 = self.convlstm3(encoded_skips[2])
        
        
        return [lstm1,lstm2,lstm3]
        


    def init_hidden_states(self):
        self.convlstm1.hidden_state = self.convlstm1._init_hidden()
        self.convlstm2.hidden_state = self.convlstm2._init_hidden()
        self.convlstm3.hidden_state = self.convlstm3._init_hidden()

