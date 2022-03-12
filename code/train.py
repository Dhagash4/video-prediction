import os,datetime
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import yaml
import click

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


from utils.visualizations import *
from utils.TrainerBaseline import *
from models.baselineLSTM import predictor as lstm
from models.vgg_baseline import *
from utils.utils import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default='configs/config.yaml')


def main(config):

    """loading configuration"""

    cfg = yaml.safe_load(open(config))
    batch_size = cfg['train']['batch_size']
    logging_path = os.path.join(cfg['logger']['tblogs'],f"{cfg['experiment']['id']}_{cfg['experiment']['embedding']}_{cfg['experiment']['predictor']}")
    saving_path = os.path.join(cfg['logger']['checkpoints'],f"{cfg['experiment']['id']}_{cfg['experiment']['embedding']}_{cfg['experiment']['predictor']}")
    model_resume = cfg['train']['resume_point']
    optimizer = cfg['train']['optimizer']
    beta1 = cfg['train']['beta1']


    """Model configurations"""

    mode = cfg['architecture']['lstm']['mode']
    num_layers = cfg['architecture']['lstm']['num_layers']

    """Loding models"""

    if (model_resume != 0) and os.path.exists(os.path.join(saving_path,f"model_{model_resume}")):
        saved_model = torch.load(os.path.join(saving_path,f"model_{model_resume}"))
        encoder = saved_model['encoder']
        decoder = saved_model['decoder']
        predictor = saved_model['predictor']
        cfg = saved_model['config']
        optimizer = cfg['optimizer']
        print(f"continuing from {model_resume}")
        
    else:

        encoder = VGGEncoder()
        decoder = VGGDecoder()
        predictor = lstm(batch_size=batch_size,mode=mode,num_layers=num_layers,device=device)
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path,exist_ok=True)
    if not os.path.exists(logging_path):

        os.makedirs(logging_path, exist_ok=True)
    
    shutil.rmtree(logging_path)
    # shutil.rmtree(saving_path)
    """optimizers"""

    if optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % optimizer)

    """loding datasets"""
    train_loader,val_loader,test_loader = load_dataset(cfg=cfg)


    """training and logging"""
    trainer = TrainerBase(device=device,
                          config=cfg,
                          encoder=encoder,
                          decoder=decoder,
                          predictor=predictor,
                          optimizer=optimizer,
                          save_path=saving_path,
                          writer=logging_path)

    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()