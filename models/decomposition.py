import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from utils.RevIN import RevIN

class Decomposition(nn.Module):
    def __init__(self,
                 input_length = [], 
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 no_decomposition = [],
                 use_amp = []):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5
        
        self.dwt = DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp)
        self.idwt = DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp)

        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [self.input_length] # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [self.pred_length] # required length of the pred seq after decom
        
        self.tfactor = tfactor
        self.dfactor = dfactor
        #################################
        self.affine = False
        self.rev_ins_normalization = False
        #################################
        
        if self.affine:
            self._init_params()
        if self.rev_ins_normalization:
            self.revin = nn.ModuleList([RevIN(self.channel) for i in range(self.level + 1)])
            
    def transform(self, x):
        # input: x shape: batch, channel, seq
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, [] # no decompose: returning the same value in yl
        return yl, yh
    
    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl # no decompose: returning the same value in x
        return x
           
    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l
    
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))
    
    def _wavelet_decompose(self, x):
        # input: x shape: batch, channel, seq
        yl, yh = self.dwt(x)
        
        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl * self.affine_weight[0]
            yl = yl + self.affine_bias[0]
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ * self.affine_weight[i + 1]
                yh_ = yh_ + self.affine_bias[i + 1]
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'norm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'norm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
        return yl, yh
    
    def _wavelet_reverse_decompose(self, yl, yh):
        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl - self.affine_bias[0]
            yl = yl / (self.affine_weight[0] + self.eps)
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ - self.affine_bias[i + 1]
                yh_ = yh_ / (self.affine_weight[i + 1] + self.eps)
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'denorm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'denorm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
                
        x = self.idwt((yl, yh))
        return x # shape: batch, channel, seq
