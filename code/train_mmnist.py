import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

from models.dcgan import *
from models.lstm import *
from data.MMNIST.mmnist import *
from utils.visualizations import *
from utils.TrainerFP import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mmnist_data_dir = "data/MMNIST/" 
train_loader, val_loader,test_loader = MMNIST(mmnist_data_dir, seq_first = True, device=device)

trainer = TrainerFP(device=device)
trainer.train(train_loader, val_loader,test_loader, device = device)