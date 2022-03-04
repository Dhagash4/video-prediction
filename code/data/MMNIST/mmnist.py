import numpy as np
import os
import wget
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def MMNIST(data_dir, batch_size = 40, seq_first=True, device = "cpu"):
    
    #dowload data if not downloaded
    file_path = os.path.join(data_dir, 'mnist_test_seq.npy')
    if not os.path.isfile(file_path):
        url = 'https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
        wget.download(url, out=data_dir)
    
    # Load Data as Numpy Array
    MovingMNIST = np.load(file_path).transpose(1, 0, 2, 3) #batch first
    
    batch_size=batch_size

    # Train, Test, Validation splits
    train_data = MovingMNIST[:8000]         
    val_data = MovingMNIST[8000:9000]       
    test_data = MovingMNIST[8000:10000]    

    def collate(batch):

        # Add channel dim, scale pixels between 0 and 1, 
        batch = torch.tensor(np.array(batch)).unsqueeze(1)     
        batch = batch / 255.0
        if seq_first:
            batch = batch.permute(2,0,1,3,4)
                     
        return batch 


    # Training Data Loader
    train_loader = DataLoader(train_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate)

    # Validation Data Loader
    val_loader = DataLoader(val_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate)                        

    # Test Data Loader
    test_loader = DataLoader(test_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate)
    
    return train_loader, val_loader, test_loader