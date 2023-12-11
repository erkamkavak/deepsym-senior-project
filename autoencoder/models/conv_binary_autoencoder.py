import torch
import torch.nn as nn

from config import INPUT_SIZE
from .model_utils import Encoder, MLP


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

        self.mlp = MLP([hidden_channel_sizes[-1], hidden_channel_sizes[-1], hidden_channel_sizes[-1]])
        self.binarizer = StraightThrough.apply

        decoder_layers = []
        for i in range(len(hidden_channel_sizes) - 1, 0, -1): 
            decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[i], hidden_channel_sizes[i-1], 
                                                     kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], input_channel_size, 
                                                 kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
    
        self.decoder = nn.Sequential(*decoder_layers)

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
        
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.mlp(x) # B, H, W, C
        x = self.binarizer(x) # B, H, W, C
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        x = self.decoder(x)
        return x

    def loss_function(self, ground_truth, output):
        return nn.MSELoss()(ground_truth, output)