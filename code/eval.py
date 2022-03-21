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
import random
import argparse
import lpips

# from utils.visualizations import *
# from utils.TrainerBaseline import *
# from models.baselineLSTM import predictor as lstm
# from models.resnet_baseline import Resnet18Encoder, Resnet18Decoder
# from models.dcgan_baseline import DCGANEncoder, DCGANDecoder
# from models.vgg_baseline import *
from utils.utils import eval_dataset,set_random_seed
from model_eval.metrices import eval_seq


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--dataset', default='MMNIST', help='dataset name')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--seed', default=3292666, type=int, help='manual seed')
parser.add_argument('--past_frames', type=int, default=10, help='# context frames')
parser.add_argument('--future_frames', type=int, default=10, help='# predicted frames')

opt = parser.parse_args()
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = opt.batch_size
saved_model = torch.load(opt.model_path,map_location=device)

encoder = saved_model['encoder']
decoder = saved_model['decoder']
predictor = saved_model['predictor']
cfg = saved_model['config']
skip = cfg['architecture']['skip']
predictor.eval()
encoder.eval()
decoder.eval()
predictor.convlstm1.batch_size = opt.batch_size
predictor.convlstm2.batch_size = opt.batch_size
predictor.convlstm3.batch_size = opt.batch_size


"""loading datasets"""

test_loader = eval_dataset(dataset=opt.dataset,batch_size=opt.batch_size)

"""setting random seed"""

set_random_seed(random_seed=opt.seed)

@torch.no_grad()
def generate_future_sequences(test_batch,skip=False):

    """ Generating a bunch of images using current state of generator """
    
    predictor.init_hidden_states()
    gt_seq = []
    pred_seq = []
    x_input = test_batch[0]
    for i in range(1, opt.past_frames+opt.future_frames):
        
        encoded_skips = encoder(x_input)
            
        if i < opt.past_frames:
            
            predictor(encoded_skips)
            x_input = test_batch[i]
            

        else:

            lstm_outputs = predictor(encoded_skips)
            if not skip:

                x_input = decoder(lstm_outputs)
            else:
                x_input = decoder([encoded_skips,lstm_outputs])
            gt_seq.append(test_batch[i])
            pred_seq.append(x_input)
    
    
    gt_seq = torch.stack(gt_seq)
    pred_seq = torch.stack(pred_seq)
    
    return gt_seq,pred_seq



"""generating sequences"""

eval_sequences = []

ssim = []
lpips_ = []
psnr= []
mse = []
mae = []
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

for i,seqs in tqdm(enumerate(test_loader),total =len(test_loader)):
    
    seqs = seqs.to(device)

    gt_seq,pred_seq = generate_future_sequences(seqs,skip=skip)
    
    lpips_seq, ssim_seq, psnr_seq, mse_seq, mae_seq = eval_seq(gt_seq,pred_seq,device,loss_fn_vgg=loss_fn_vgg)
    lpips_.append(lpips_seq)
    ssim.append(ssim_seq)
    psnr.append(psnr_seq)
    mse.append(mse_seq)
    mae.append(mae_seq)
    eval_sequences.append([seqs.cpu().numpy(),np.concatenate((gt_seq.cpu().numpy(),pred_seq.cpu().numpy()),axis=0)])

# eval_sequences = torch.stack(eval_seq)
lpips_ = np.concatenate((lpips_))
ssim   = np.concatenate((ssim))
mse    = np.concatenate((mse))
mae    = np.concatenate((mae))
psnr   = np.concatenate((psnr))


