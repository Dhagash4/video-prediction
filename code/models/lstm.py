import torch
import torch.nn as nn
import torchvision

class predictor_lstm(nn.Module):
    """ 
    Sequential classifier for images. Embedded image rows are fed to a RNN
    
    Args:
    -----
    input_dim: integer
        dimensionality of the rows to embed
    out_dim: integer 
        dimensionality of the output LSTM
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        number of stacked LSTMS
    mode: string
        intialization of the states
    """
    
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers=1, mode="zeros", batch_size =100):
        """ Module initializer """
        assert mode in ["zeros", "random", "learned"]
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim =  hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.mode = mode
        self.batch_size = batch_size
       
        # for embedding rows into vector representations
        self.enc = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        # LSTM model
        self.lstm = nn.ModuleList([nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim) for i in range(self.num_layers)])
        
        # FC-classifier
        self.out = nn.Sequential(
              nn.Linear(in_features=hidden_dim, out_features=out_dim),
              nn.Tanh())
        
        return
    
    
    def forward(self, x):
        """ Forward pass through model """
        
        # b_size, _ = x.shape
        h, c = self.init_state(b_size=self.batch_size, device=x.device) 
        
        # embedding rows
        # x_rowed = x.view(b_size, n_channels*n_rows, n_cols)
        # x_rowed = x.view(-1, self.input_dim)
        embeddings = self.enc(x.view(-1, self.input_dim))

        # feeding LSTM. Does everything for you
        # lstm_out, (h_out, c_out) = self.lstm(embeddings, (h,c)) 

        # feeding LSTM. Does everything for you
        h_input = embeddings
        for i in range(self.num_layers):
            h[i], c[i] = self.lstm[i](h_input.clone(), (h[i].clone(),c[i].clone()))
            h_input = h[i].clone()
        
        # classifying
        # y = self.out(lstm_out[:, -1, :])  # feeding only output at last layer
        y = self.out(h_input)
        
        return y
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(self.num_layers, b_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, b_size, self.hidden_dim).to(device)
        elif(self.mode == "random"):
            h = torch.randn(self.num_layers, b_size, self.hidden_dim).to(device)
            c = torch.randn(self.num_layers, b_size, self.hidden_dim).to(device)
        # elif(self.mode == "learned"):
        #     h = self.learned_h.repeat(1, b_size, 1)
        #     c = self.learned_c.repeat(1, b_size, 1)
        # h = h.clone().to(device)
        # c = c.clone().to(device)
        return h, c


class latent_lstm(nn.Module):
    """ 
    Sequential classifier for images. Embedded image rows are fed to a RNN
    
    Args:
    -----
    input_dim: integer
        dimensionality of the rows to embed
    out_dim: integer 
        dimensionality of the output LSTM
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        number of stacked LSTMS
    mode: string
        intialization of the states
    """
    
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers=1, mode="zeros", batch_size=100):
        """ Module initializer """
        assert mode in ["zeros", "random", "learned"]
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim =  hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.mode = mode
        self.batch_size = batch_size
       
        # for embedding rows into vector representations
        self.enc = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        # LSTM model
        self.lstm = nn.ModuleList([nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim) for i in range(self.num_layers)])
        
        # FC-classifier
        self.mu = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.log_var =  nn.Linear(in_features=hidden_dim, out_features=out_dim)
        
        return
    

    def reparameterize(self, mu, log_var):
        """ Reparametrization trick"""
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)  # random sampling happens here
        z = mu + std * eps
        return z
    
    def forward(self, x):
        """ Forward pass through model """
        h, c = self.init_state(b_size=self.batch_size, device=x.device) 
        
        # embedding rows
        # x_rowed = x.view(b_size, n_channels*n_rows, n_cols)
        # x_rowed = x.view(-1, self.input_dim)
        embeddings = self.enc(x.view(-1, self.input_dim))
#         embeddings = embeddings.view(b_size, n_channels*n_rows, n_cols)

        # feeding LSTM. Does everything for you
        h_input = embeddings
        for i in range(self.num_layers):
            h[i], c[i] = self.lstm[i](h_input.clone(), (h[i].clone(),c[i].clone()))
            h_input = h[i].clone()
                                    
        # classifying
        # y = self.o  ut(lstm_out[:, -1, :])  # feeding only output at last layer
        # y = self.out(h_out)
        mu = self.mu(h_input)
        log_var = self.log_var(h_input)
        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(self.num_layers, b_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, b_size, self.hidden_dim).to(device)
        elif(self.mode == "random"):
            h = torch.randn(self.num_layers, b_size, self.hidden_dim).to(device)
            c = torch.randn(self.num_layers, b_size, self.hidden_dim).to(device)
        # elif(self.mode == "learned"):
        #     h = self.learned_h.repeat(1, b_size, 1)
        #     c = self.learned_c.repeat(1, b_size, 1)
        # h = h.to(device)
        # c = c.to(device)
        return h, c