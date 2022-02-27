import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MMNIST(batch_size = 40, seq_first="True", device = "cpu"):
    
    
    #dowload data if not downloaded
    #! wget -q https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
    
    # Load Data as Numpy Array
    MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3) #batch first

    batch_size=40
#     past_frames = 10
#     future_frames = 10

    # Train, Test, Validation splits
    train_data = MovingMNIST[:8000]         
#     val_data = MovingMNIST[8000:9000]       
    test_data = MovingMNIST[8000:10000]    

    def collate(batch):

        # Add channel dim, scale pixels between 0 and 1, 
        batch = torch.tensor(batch).unsqueeze(1)     
        batch = batch / 255.0
        x = batch[0].squeeze(0)
        x = x.unsqueeze(1)
        x= x.unsqueeze(2)
        # x.shape
        for y in batch[1:]:
            y = y.squeeze(0)
            y = y.unsqueeze(1)
            y= y.unsqueeze(2)
            x = torch.cat([x,y], 1)

        x = x.to(device)                     

        return x
        # Randomly pick 10 frames as input, 11th frame is target
        # rand = np.random.randint(10,20)                     
        # return batch[:,:,:10], batch[:,:,10:]     


    # Training Data Loader
    train_loader = DataLoader(train_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate)

    # Validation Data Loader
    test_loader = DataLoader(test_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate)
    
    return train_loader, test_loader