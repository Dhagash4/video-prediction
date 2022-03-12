import os,datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

from data.MMNIST.mmnist import *
from utils.visualizations import *
from data.MMNIST.moving_mnist import *
from utils.TrainerBaseline import *
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = "MMNIST"
data_dir = "data"

TRAINING_LOGS = os.path.join(os.getcwd(), "tboard_logs", f"{data}_{datetime.datetime.now().strftime('%Y-%m-%d')}")
MODEL_SAVE = os.path.join(os.getcwd(), "checkpoints", f"{data}_{datetime.datetime.now().strftime('%Y-%m-%d')}")


if not os.path.exists(TRAINING_LOGS):
    os.makedirs(TRAINING_LOGS)
else:
    shutil.rmtree(TRAINING_LOGS)

if not os.path.exists(MODEL_SAVE):
    os.makedirs(MODEL_SAVE)
#else:
 #   shutil.rmtree(MODEL_SAVE)
mmnist_data_dir = os.path.join(data_dir,data)

test_loader = MMNIST(mmnist_data_dir, seq_first = True, device=device,batch_size=20)
train_data= MovingMNIST(train=True,data_root=mmnist_data_dir,seq_len=20)
test_data = MovingMNIST(train=False,data_root=mmnist_data_dir,seq_len=20)
train_loader = DataLoader(train_data,
                          num_workers=0,
                          batch_size=20,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
val_loader = DataLoader(test_data,
                         num_workers=0,
                         batch_size=20,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)   
trainer = TrainerBase(device=device,writer=TRAINING_LOGS,save_path=MODEL_SAVE,batch_size=20)
trainer.train(train_loader, test_loader, test_loader, device = device,num_epochs=50)
