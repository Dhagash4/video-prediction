import os
import shutil
import yaml
import click
import torch

from utils.Trainer import TrainerBase
from models.predictorLSTM import predictor as lstm
from models.resnet import Resnet18Encoder, Resnet18Decoder
from models.dcgan import DCGANEncoder, DCGANDecoder
from models.vgg import VGGEncoder, VGGDecoder
from utils.utils import load_dataset,set_random_seed

@click.command()
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
    scheduler = cfg['train']['scheduler']
    beta1 = cfg['train']['beta1']
    embedding = cfg['experiment']['embedding']
    skip_connection = cfg['architecture']['skip']
    gpu_num = cfg['train']['device']

    
    """Model configurations"""

    mode = cfg['architecture']['lstm']['mode']
    num_layers = cfg['architecture']['lstm']['num_layers']
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    random_seed = cfg['random_seed']
 
    """set random seed"""
    set_random_seed(random_seed=random_seed)

    """print configuration"""

    print(f"Configuration is:\n{cfg}")

    """Loding models"""
    if (model_resume != 0) and os.path.exists(os.path.join(saving_path,f"model_{model_resume}.pth")):
        saved_model = torch.load(os.path.join(saving_path,f"model_{model_resume}.pth"))
        encoder = saved_model['encoder']
        decoder = saved_model['decoder']
        predictor = saved_model['predictor']
        cfg = saved_model['config']
        optimizer = cfg['train']['optimizer']
        
        print(f"continuing from {model_resume}")
        
    else:
        if embedding == "vgg":
            
            encoder = VGGEncoder()
            decoder = VGGDecoder(skip_connection=skip_connection)
            decoder = VGGDecoder()
        elif embedding == "resnet":
            encoder = Resnet18Encoder()
            decoder = Resnet18Decoder(skip_connection=skip_connection)
            # decoder = Resnet18Decoder()
        
        elif embedding == "dcgan":

            encoder = DCGANEncoder()
            decoder = DCGANDecoder(skip_connection=skip_connection)

        else:

            print(f"[ERROR] Not valid embedding layer")

        predictor = lstm(batch_size=batch_size,mode=mode,num_layers=num_layers,device=device)
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path,exist_ok=True)
    if not os.path.exists(logging_path):

        os.makedirs(logging_path, exist_ok=True)
    
    shutil.rmtree(logging_path)

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
                          scheduler=scheduler,
                          save_path=saving_path,
                          writer=logging_path,resume_point=model_resume)

    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
