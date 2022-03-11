from __future__ import print_function, division

import os
import pickle
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class KTH(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, data_dir, image_size=128, train = True, val = False, seq_len=20, frame_skip = 20, transform=None):
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

        # self.data= {}
        if train:
            self.train = True
            data_type = 'train'
            self.videos = pickle.load(open(os.path.join(data_dir,"train.p"), "rb")) 
        elif val:
            self.train = False
            data_type = 'train'
            self.videos = pickle.load(open(os.path.join(data_dir,"val.p"), "rb"))
        else:
            self.train = False
            self.videos = pickle.load(open(os.path.join(data_dir,"test.p"), "rb")) 
            data_type = 'test'
                
        #for all sequences
        self.sequences, self.labels = self.create_sequences()
        
        # for infinite generation
#         for c in self.classes:
#             self.data[c] = []
#         for i in range(len(self.videos)):
#             category = self.videos[i]["category"] 
#             self.data[category].append(self.videos[i]["frames"])
    
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
        while True: # skip seqeunces that are too short
            c_idx = np.random.randint(len(self.classes))
            c = self.classes[c_idx]
            vid_idx = np.random.randint(len(self.data[c]))
            vid = self.data[c][vid_idx]
            seq_idx = np.random.randint(len(vid))
            if seq_idx - t >= 0:
                break
#         dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        st = random.randint(0, seq_idx-t)


        seq = [] 
        for i in range(st, st+t):
#             fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
#             im = misc.imread(fname)/255.
            im = vid[i]
            seq.append(im.reshape(self.image_size, self.image_size, 1))
        return np.array(seq)

    def __getitem__(self, index):
        sample = { 
            "seq": torch.from_numpy(self.sequences[index]), 
            "label": self.labels[index] 
        }

        return sample          
                   
#         if not self.seed_set:
#             self.seed_set = True
#             random.seed(index)
#             np.random.seed(index)
#             #torch.manual_seed(index)
#         return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return len(self.sequences)

def get_KTH(data_dir, batch_size = 40, seq_first=True, frame_skip=20, device = "cpu"):
    
    
    batch_size=batch_size

    # Train, Test, Validation splits
    train_data = KTH(data_dir, train = True, val = False, frame_skip=frame_skip)       
    val_data = KTH(data_dir, train = False, val = True)        
    test_data = KTH(data_dir, train = False, val = False)       

    def collate(batch):

        # Add channel dim, scale pixels between 0 and 1, 
        # batch = torch.tensor(batch).unsqueeze(1)
        labels = []
        seqs = []
        for i in range(len(batch)):
            batch[i]['seq'] = torch.tensor(batch[i]['seq']) / 255.0
            batch[i]['seq'] = batch[i]['seq'].permute(0,3,1,2)
            batch[i]['seq'] = batch[i]['seq'].to(device)
            seqs.append(batch[i]['seq'])
            labels.append(batch[i]['label'])
            # batch[i]['label'] = batch[i]['label'].to(device)    
        seqs = torch.stack(seqs)
        if seq_first:
            seqs = seqs.permute(1,0,2,3,4)    
        return (seqs, labels)   
        
    # Training Data Loader
    train_loader = DataLoader(train_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True)

    # Validation Data Loader
    val_loader = DataLoader(val_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True)                        

    # Test Data Loader
    test_loader = DataLoader(test_data, shuffle=True, 
                            batch_size=batch_size, collate_fn=collate, drop_last=True)
    
    return train_loader, val_loader, test_loader