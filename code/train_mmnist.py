import os,datetime
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
data = "MMNIST"
data_dir = "data"

TRAINING_LOGS = os.path.join(os.getcwd(), "tboard_logs", f"{data}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

if not os.path.exists(TRAINING_LOGS):
    os.makedirs(TRAINING_LOGS)

mmnist_data_dir = os.path.join(data_dir,data)

train_loader, val_loader,test_loader = MMNIST(mmnist_data_dir, seq_first = True, device=device)

trainer = TrainerFP(device=device,writer=TRAINING_LOGS)
trainer.train(train_loader, val_loader,test_loader, device = device)