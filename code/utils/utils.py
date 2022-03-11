import os
import torch
from torch.autograd import Variable
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
        
        if cfg['dataset'] == 'MMNIST':
                mmnist_data_dir = os.path.join(ROOT_DIR,cfg['dataset'])
                
                if not os.path.exists(mmnist_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                train_loader, val_loader, test_loader = MMNIST(mmnist_data_dir, batch_size=cfg['train']['batch_size'],seq_first = True)
        else:
                kth_data_dir = os.path.join(ROOT_DIR,cfg['dataset'])
                
                if not os.path.exists(kth_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                train_loader, val_loader, test_loader  = get_KTH(kth_data_dir, batch_size = cfg['train']['batch_size'], seq_first=True, frame_skip=cfg['dataset']['seq'])

        return train_loader, val_loader, test_loader                 



