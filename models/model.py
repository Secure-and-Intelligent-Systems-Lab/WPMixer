import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import Permute, Reshape
from utils.RevIN import RevIN

import matplotlib.pyplot as plt
import numpy as np
from models.wavelet_patch_mixer import WPMixerCore

class WPMixerWrapperShortTermForecast(nn.Module):
    def __init__(self,
                 c_in = [], 
                 c_out = [],
                 seq_len = [],
                 out_len = [], 
                d_model = [],  
                dropout = [], 
                embedding_dropout = [],
                device = [],
                batch_size = [],
                tfactor = [],
                dfactor = [],
                wavelet = [],
                level = [],
                patch_len = [],
                stride = [],
                no_decomposition = [],
                use_amp = []):
        super(WPMixerWrapperShortTermForecast, self).__init__()
        self.model = WPMixer(c_in = c_in, c_out = c_out, seq_len = seq_len, out_len = out_len, d_model = d_model,
                            dropout = dropout, embedding_dropout = embedding_dropout, device = device, batch_size = batch_size,
                            tfactor = tfactor, dfactor = dfactor, wavelet = wavelet, level = level, patch_len = patch_len,
                            stride = stride, no_decomposition = no_decomposition,
                            use_amp = use_amp)
        
    def forward(self, x, _unknown1, _unknown2, _unknown3):
        out = self.model(x)
        return out
    

class WPMixer(nn.Module):
    def __init__(self,
                 c_in = [], 
                 c_out = [],
                 seq_len = [],
                 out_len = [], 
                d_model = [],  
                dropout = [], 
                embedding_dropout = [],
                device = [],
                batch_size = [],
                tfactor = [],
                dfactor = [],
                wavelet = [],
                level = [],
                patch_len = [],
                stride = [],
                no_decomposition = [],
                use_amp = []):
        
        super(WPMixer, self).__init__()
        self.pred_len = out_len
        self.channel_in = c_in
        self.channel_out = c_out
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_size = batch_size # not required now
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.wavelet = wavelet
        self.level = level
        # patch predictior
        self.actual_seq_len = seq_len
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.device = device
        
        self.wpmixerCore = WPMixerCore(input_length = self.actual_seq_len,
                                                      pred_length = self.pred_len,
                                                      wavelet_name = self.wavelet,
                                                      level = self.level,
                                                      batch_size = self.batch_size,
                                                      channel = self.channel_in, 
                                                      d_model = self.d_model, 
                                                      dropout = self.dropout, 
                                                      embedding_dropout = self.embedding_dropout,
                                                      tfactor = self.tfactor, 
                                                      dfactor = self.dfactor, 
                                                      device = self.device,
                                                      patch_len = self.patch_len, 
                                                      patch_stride = self.stride,
                                                      no_decomposition = self.no_decomposition,
                                                      use_amp = self.use_amp)
        
        
    def forward(self, x):
        pred = self.wpmixerCore(x)
        pred = pred[:, :, -self.channel_out:]
        return pred 

