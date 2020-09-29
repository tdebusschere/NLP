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

class Highway(nn.Module):
    """ 
        - Achieving Open Vocabulary Neural Machine Translation (Luong, et al. 2016/06)
    """
    def __init_(self, word_embed_size):
        super().__init__()
        self.Wproj = nn.linear( word_embed_size, word_embed_size, bias=True)
        self.Wgate = nn.Linear( word_embed_size, word_embed_size, bias=True)
    
    """ transforming  the layer result into a result"""
    def forward(self, inputval: torch.Tensor):
        """" 
\        """
        x_proj = F.relu( self.Wproj( inputval ))
        x_gate = torch.sigmoid( self.Wgate( inputval ))
    
        x_high = x_gate * xproj + (1 - x_gate) * inputval
        return( x_high )
