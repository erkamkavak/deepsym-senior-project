import torch
import torch.nn as nn

from config import INPUT_SIZE
from .model_utils import Encoder, build_decoder


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)


class ConvBinaryFeatureAutoEncoder(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes, pretrained_model=""): 
        super(ConvBinaryFeatureAutoEncoder, self).__init__() 

        self.encoder = Encoder(input_channel_size, hidden_channel_sizes)
        self.binarizer = StraightThrough.apply

        self.decoder = build_decoder(hidden_channel_sizes, input_channel_size) 
        
        if pretrained_model != "": 
            self.load_pretrained_model(pretrained_model)

    def load_pretrained_model(self, model_path): 
        pretrained_dict = torch.load(model_path)
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(pretrained_dict)

    def forward(self, x):
        x = self.encoder(x)        
        x = self.binarizer(x)
        x = self.decoder(x)
        return x

    def loss_function(self, ground_truth, output):
        return nn.MSELoss()(ground_truth, output)