import math
import torch
import torch.nn as nn

from config import INPUT_SIZE


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)
    


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_dims, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_dims[0]
        for i, unit in enumerate(layer_dims[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_dims[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class Encoder(nn.Module): 
    previously_saved_path = None

    def __init__(self, input_channel_size, hidden_channel_sizes) -> None:
        super(Encoder, self).__init__()        
        dimension_reduction = 2 ** len(hidden_channel_sizes)
        if INPUT_SIZE[0] % dimension_reduction != 0: 
            raise ValueError("Input size must be divisible by 2^num_hidden_layers")

        layers = [
            nn.Conv2d(input_channel_size, hidden_channel_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(True),
        ]
        for i in range(1, len(hidden_channel_sizes)): 
            layers.append(nn.Conv2d(hidden_channel_sizes[i-1], hidden_channel_sizes[i], 
                                            kernel_size=3, padding=1))
            layers.append(nn.ReLU(True))
            layers.append(nn.Conv2d(hidden_channel_sizes[i], hidden_channel_sizes[i], 
                                            kernel_size=3, stride=2, padding=1))
            # encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder = nn.Sequential(*layers)
        self.mlp = MLP([hidden_channel_sizes[-1], hidden_channel_sizes[-1], hidden_channel_sizes[-1]])

    def forward(self, x): 
        x = self.encoder(x) # B, C, H, W
        
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.mlp(x) # B, H, W, C
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        return x
    
    def save_model(self, model_name): 
        torch.save(self.state_dict(), f"{model_name}-encoder.pt")
        Encoder.previously_saved_path = f"{model_name}-encoder.pt"
    
    @staticmethod
    def load_model(input_channel_size, hidden_channel_sizes, model_name=None): 
        load_path = ""
        if model_name == None: 
            if Encoder.previously_saved_path == None: 
                raise Exception("Encoder model can not be loaded!")
            load_path = Encoder.previously_saved_path
        else: 
            load_path = f"{model_name}-encoder.pt"
        encoder = Encoder(input_channel_size, hidden_channel_sizes)
        encoder.load_state_dict(torch.load(load_path))
        return encoder
    

def build_decoder(hidden_channel_sizes, last_layer_channel_size): 
    decoder_layers = []
    for i in range(len(hidden_channel_sizes) - 1, 0, -1): 
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[i], hidden_channel_sizes[i-1], 
                                                    kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.ReLU(True))
    
    decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], last_layer_channel_size, 
                                                kernel_size=3, padding=1))
    decoder_layers.append(nn.Tanh())

    decoder = nn.Sequential(*decoder_layers)
    return decoder