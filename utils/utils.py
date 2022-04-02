"""credits: Angel Villar-Corrales, https://github.com/edenton/svg"""
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.MMNIST.mmnist import MMNIST
from data.MMNIST.moving_mnist import MovingMNIST
from data.KTH.kth import get_KTH
import tensorflow as tf
import random
import numpy as np

def sequence_input(seq, dtype):
        return [Variable(x.type(dtype)) for x in seq]

def normalize_data(dtype, sequence):
    
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    
    return sequence_input(sequence, dtype)

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
    
def set_random_seed(random_seed):
    
    """
    Using random seed for numpy and torch
    """
   
    random_seed = random_seed
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def torch_to_tf(torch_tensor: torch.FloatTensor) -> tf.Tensor:
    torch_tensor = torch_tensor.permute([1, 0, 3, 4, 2]).expand(-1,-1,-1,-1,3)  # channels last
    np_tensor = torch_tensor.detach().cpu().numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor

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


def eval_dataset(dataset="MMNIST",num_workers=4,seq_len=20,batch_size = 16):
        
        ROOT_DIR = 'data/'
        

        if dataset == 'MMNIST':
                mmnist_data_dir = os.path.join(ROOT_DIR,dataset)
                
                if not os.path.exists(mmnist_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                test_loader, _ = MMNIST(mmnist_data_dir, batch_size=batch_size,seq_first = True,num_workers=num_workers)
                

                return test_loader

        elif dataset == 'KTH':
                kth_data_dir = os.path.join(ROOT_DIR,dataset)
                
                if not os.path.exists(kth_data_dir):
                        
                        print(f"[ERROR] Directory dosent exists please ensure data is in data folder")
                
                _, _, test_loader  = get_KTH(kth_data_dir, batch_size = batch_size, seq_first=True, num_workers=num_workers)

                return test_loader                



def get_data_batch(cfg, train_loader,dtype=torch.FloatTensor):
        while True:
            for sequence in train_loader:
                if cfg['data']['dataset'] == "MMNIST":
                        batch = normalize_data(dtype, sequence)
                        batch = torch.stack(batch)
                        yield batch
                elif cfg['data']['dataset'] == "KTH":
                        yield sequence