#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import random

class CNN(nn.Module):


### YOUR CODE HERE for part 1i
    """ 
        - Achieving Open Vocabulary Neural Machine Translation (Luong, et al. 2016/06)
        - input 
    """
    def __init__(self, char_embed_size, word_embed_size, kernel_size ):
        super(CNN, self).__init__()
        self.convmask = nn.Conv1d( in_channels = char_embed_size, out_channels = word_embed_size, kernel_size = kernel_size)
    
    """ transforming  the layer result into a result"""
    def forward(self, inputval: torch.Tensor):
        """" 
\        """
        x_proj = torch.max(F.relu( self.convmask( inputval )), dim = 2)[0]
        return( x_proj )


### END YOUR CODE

