import torch
from torch.autograd import Variable

def sequence_input(seq, dtype):
        return [Variable(x.type(dtype)) for x in seq]

def normalize_data(dtype, sequence):
    
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    
    return sequence_input(sequence, dtype)