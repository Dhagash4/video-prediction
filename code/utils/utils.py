import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.dcgan import *
from models.lstm import *
from data.MMNIST.mmnist import *
from utils.visualizations import *
from utils.TrainerFP import *
from data.MMNIST.moving_mnist import *
from data.KTH.kth import *

def sequence_input(seq, dtype):
        return [Variable(x.type(dtype)) for x in seq]

def normalize_data(dtype, sequence):
    
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    
    return sequence_input(sequence, dtype)


def load_dataset(cfg):
        
        ROOT_DIR = 'data/'
        num_workers = cfg['train']['num_workers']
        seq_len = cfg['data']['seed_frames'] + cfg['data']['predict_frames'] 
        batch_size = cfg['train']['batch_size']

        if cfg['data']['dataset'] == 'MMNIST':
                mmnist_data_dir = os.path.join(ROOT_DIR,cfg['data']['dataset'])
                
                if not os.path.exists(mmnist_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                test_loader, val_loader = MMNIST(mmnist_data_dir, batch_size=batch_size,seq_first = True,num_workers=num_workers)
                train_data= MovingMNIST(train=True,data_root=mmnist_data_dir,seq_len=seq_len)
                train_loader = DataLoader(train_data,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

                

                return train_loader, val_loader, test_loader

        elif cfg['data']['dataset'] == 'KTH':
                kth_data_dir = os.path.join(ROOT_DIR,cfg['data']['dataset'])
                
                if not os.path.exists(kth_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                train_loader, val_loader, test_loader  = get_KTH(kth_data_dir, batch_size = batch_size, seq_first=True, num_workers=num_workers)

                return train_loader, val_loader, test_loader                 



def get_data_batch(cfg, train_loader, dtype=torch.cuda.FloatTensor):
        while True:
            for sequence in train_loader:
                if cfg['data']['dataset'] == "MMNIST":
                        batch = normalize_data(dtype, sequence)
                        batch = torch.stack(batch)
                        yield batch
                elif cfg['data']['dataset'] == "KTH":
                        yield sequence


def get_testing_batch(test_loader,dtype=torch.cuda.FloatTensor):
        while True:
                for sequence in test_loader:
                        batch = normalize_data(dtype, sequence)
                        batch = torch.stack(batch)
                        yield batch 