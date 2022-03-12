from tkinter import image_names
import torch
import torch.nn as nn
import torchvision

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, image_size = (128,8,8), mode="zeros", device="cpu"):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.mode = mode
        self.device = device
        self.image_size = image_size

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.W_ci = nn.Parameter(torch.zeros(1, self.hidden_dim,  self.image_size[1], self.image_size[2]))
        self.W_cf = nn.Parameter(torch.zeros(1, self.hidden_dim,  self.image_size[1], self.image_size[2]))
        self.W_co = nn.Parameter(torch.zeros(1, self.hidden_dim,  self.image_size[1], self.image_size[2]))
        

    def forward(self, x, cur_state):
        
        h_cur, c_cur = cur_state
        x = x.to(self.device)
        h_cur = h_cur.to(self.device)

        concat_input_hcur = torch.cat([x, h_cur], dim=1) 
        concat_input_hcur = concat_input_hcur.to(self.device)

        concat_input_hcur_conv = self.conv(concat_input_hcur)
        concat_input_hcur_conv = concat_input_hcur_conv.to(self.device)

        cc_input_gate, cc_forget_gate, cc_output_gate, cc_output = torch.split(concat_input_hcur_conv, self.hidden_dim, dim=1)
        
        input_gate = torch.sigmoid(cc_input_gate + self.W_ci * c_cur)

        forget_gate = torch.sigmoid(cc_forget_gate + self.W_cf * c_cur)

        output = torch.tanh(cc_output)

        c_next = forget_gate * c_cur + input_gate * output

        output_gate = torch.sigmoid(cc_output_gate + self.W_co * c_next)

        h_next = output * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, batch_size, image_size):
        height, width = image_size[1], image_size[2]
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "random"):
            h = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "learned"):
            h = self.learned_h.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
            c = self.learned_c.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
        
        return h, c

        

class predictor_lstm(nn.Module):
    
    def __init__(self, input_dim = (64,32,32), hidden_dim = 64, kernel_sizes = (5,5), return_all_layers = False,
                num_layers=2, mode="zeros",  batch_size = 40, bias=True, device = "cpu"):
        """ Module initializer """
        assert mode in ["zeros", "random", "learned"]
        super().__init__()

        self.num_layers = num_layers
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.return_all_layers = return_all_layers
        self.bias = bias

        if type(hidden_dim) is list:
            assert self.num_layers == len(hidden_dim)
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self._extend_for_multilayer(hidden_dim, self.num_layers)

        if type(kernel_sizes) is list:
            assert self.num_layers == len(kernel_sizes)
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = self._extend_for_multilayer(kernel_sizes, self.num_layers)
        
        self.input_dim = input_dim[0] 
        conv_lstms  = []
        self.image_size = input_dim
        # iterating over no of layers
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            conv_lstms.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_sizes[i],
                                          bias=self.bias,
                                          image_size = self.image_size,
                                          device = self.device))

        self.conv_lstms = nn.ModuleList(conv_lstms)

        self.hidden_state = self._init_hidden()

        return
    
    
    def forward(self, x):

        h_input = x
        for i in range(self.num_layers):

            self.hidden_state[i] = self.conv_lstms[i](x= h_input, cur_state = self.hidden_state[i])
            
            h,c  = self.hidden_state[i]
            h_input = h

        return h_input
        
    def _init_hidden(self):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.conv_lstms[i].init_state(self.batch_size, self.image_size))
        return init_states
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



