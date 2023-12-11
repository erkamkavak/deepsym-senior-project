import torch
import torch.nn as nn

from .model_utils import Encoder, build_decoder

class SymbolAutoEncoder(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes): 
        super(SymbolAutoEncoder, self).__init__() 

        self.feature_encoder = Encoder.load_model(input_channel_size, hidden_channel_sizes)
        self.symbol_encoder = BinarizedEncoder()

        # TODO: change input channel size for decoder
        hidden_channel_sizes[0] = hidden_channel_sizes[0]
        self.decoder = build_decoder(hidden_channel_sizes)

    def forward(self, x, action):
        features = self.feature_encoder(x) # B, C, H, W
        symbols = self.symbol_encoder(x) 

        decoder_input = torch.cat([features, symbols, action])
        
        output = self.decoder(decoder_input)
        return output

    def loss_function(self, ground_truth, output):
        return nn.MSELoss()(ground_truth, output)
    
    def save_encoder(self, save_path): 
        self.feature_encoder.save_model(save_path)



# def train_first_conv_then_symbol_autoencoders(): 
#     conv_autoencoder = 


if __name__ == "__main__":
    # image must have a size as power of 2
    test_input = torch.randn(1, 3, 64, 64)
    model = SymbolAutoEncoder(
        input_channel_size=3,
        hidden_channel_sizes=[8, 16, 32]
    )
    print(model)
    out1 = model(test_input)
    for i in range(len(model.encoder)):
        layer = model.encoder[i]
        test_input = layer(test_input)
        print(f"encoder layer {i}: {test_input.shape}")
    for layer in model.decoder:
        test_input = layer(test_input)
        print(test_input.shape)

    assert test_input.shape == (1, 3, 64, 64)