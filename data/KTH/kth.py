"""credits: https://github.com/vkhoi/KTH-Action-Recognition/blob/master/main/data_utils.py"""
from __future__ import print_function, division

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import subprocess

class KTH(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, data_dir, image_size=64, train = True, val = False, seq_len=20, frame_skip = 20, transform=None):
        """
        Args:
    
        """
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        self.image_size = image_size
        self.data_dir = data_dir
        self.frame_skip = frame_skip
        self.seq_len = seq_len
        self.transform = transform
        self.seed_set = False

        if not os.path.exists(os.path.join(data_dir,"kth")):
            subprocess.call(['sh' , './' + os.path.join(data_dir,'download_kth.sh')])
            subprocess.call(['python', os.path.join(data_dir,'preprocess_kth.py')])


        if train:
            self.train = True
            data_type = 'train'
            self.videos = pickle.load(open(os.path.join(data_dir,"kth/train.p"), "rb")) 
        elif val:
            self.train = False
            data_type = 'train'
            self.videos = pickle.load(open(os.path.join(data_dir,"kth/val.p"), "rb"))
        else:
            self.train = False
            self.videos = pickle.load(open(os.path.join(data_dir,"kth/test.p"), "rb")) 
            data_type = 'test'
                
        self.sequences, self.labels = self.create_sequences()

    
    def create_sequences(self):
        
        sequences = []
        labels = []
        for i in range(len(self.videos)):
            category = self.videos[i]["category"] 
            for j in range(0, len(self.videos[i]["frames"]),self.frame_skip):
                if(len(self.videos[i]["frames"]) - j < self.seq_len):
                    temp =  len(self.videos[i]["frames"]) - self.seq_len
                    seq = [] 
                    for k in range(temp, temp+self.seq_len):
                        im = self.videos[i]["frames"][k]
                        seq.append(im.reshape(self.image_size, self.image_size, 1))
                    sequences.append(np.array(seq))
                    labels.append(category)
                                
                else:  
                    seq = [] 
                    for k in range(j, j+self.seq_len):
                        im = self.videos[i]["frames"][k]
                        seq.append(im.reshape(self.image_size, self.image_size, 1))
                    sequences.append(np.array(seq))
                    labels.append(category)
        
        return sequences, labels
    
    def get_sequence(self):
        t = self.seq_len
        while True:
            c_idx = np.random.randint(len(self.classes))
            c = self.classes[c_idx]
            vid_idx = np.random.randint(len(self.data[c]))
            vid = self.data[c][vid_idx]
            seq_idx = np.random.randint(len(vid))
            if seq_idx - t >= 0:
                break
        st = random.randint(0, seq_idx-t)


        seq = [] 
        for i in range(st, st+t):
            im = vid[i]
            seq.append(im.reshape(self.image_size, self.image_size, 1))
        return np.array(seq)

    def __getitem__(self, index):
        sample = { 
            "seq": torch.from_numpy(self.sequences[index]), 
            "label": self.labels[index] 
        }

        return sample          

    def __len__(self):
        return len(self.sequences)

def get_KTH(data_dir, batch_size = 40, seq_first=True, frame_skip=5, device = "cpu",num_workers=4):
    
    
    batch_size=batch_size
    

    """Train, Test, Validation splits"""

    train_data = KTH(data_dir, train = True, val = False, frame_skip=frame_skip)       
    val_data = KTH(data_dir, train = False, val = True)        
    test_data = KTH(data_dir, train = False, val = False)       

    def collate(batch):

        
        labels = []
        seqs = []
        for i in range(len(batch)):
            batch[i]['seq'] = batch[i]['seq'] / 255.0
            batch[i]['seq'] = batch[i]['seq'].permute(0,3,1,2)
            batch[i]['seq'] = batch[i]['seq'].to(device)
            seqs.append(batch[i]['seq'])
            labels.append(batch[i]['label'])
            # batch[i]['label'] = batch[i]['label'].to(device)    
        seqs = torch.stack(seqs)
        if seq_first:
            seqs = seqs.permute(1,0,2,3,4)    
        return seqs  
        
    train_loader = DataLoader(train_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True,num_workers = num_workers)

    val_loader = DataLoader(val_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True,num_workers = num_workers)                        

    test_loader = DataLoader(test_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True,num_workers = num_workers)
    
    return train_loader, val_loader, test_loader
