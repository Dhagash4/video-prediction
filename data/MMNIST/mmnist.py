import numpy as np
import os
import wget
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def MMNIST(data_dir, batch_size = 40, seq_first=True, device = "cpu",num_workers=4):
    
    file_path = os.path.join(data_dir, 'mnist_test_seq.npy')
    if not os.path.isfile(file_path):
        url = 'https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        wget.download(url, out=data_dir)
    
    MovingMNIST = np.load(file_path).transpose(1, 0, 2, 3) #batch first
    
    batch_size=batch_size

  
    val_data = MovingMNIST[:1000]       
    test_data = MovingMNIST   
    

    def collate(batch):

        batch = torch.tensor(np.array(batch)).unsqueeze(1)     
        batch = batch / 255.0
        if seq_first:
            batch = batch.permute(2,0,1,3,4)
                     
        return batch 


    val_loader = DataLoader(val_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate,num_workers=num_workers)                        

    
    test_loader = DataLoader(test_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate,num_workers=num_workers)
    
    return test_loader, val_loader