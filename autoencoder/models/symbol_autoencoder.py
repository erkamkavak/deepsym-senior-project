import torch
import torch.nn as nn

from .model_utils import Encoder, build_decoder, BinarizedEncoder, convert_action_to_matrix

class SymbolAutoEncoder(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes, pretrained_encoder_name=None): 
        super(SymbolAutoEncoder, self).__init__()
        self.symbol_length = 8

        self.feature_encoder = Encoder.load_model(input_channel_size, hidden_channel_sizes, pretrained_encoder_name)
        self.symbol_encoder = BinarizedEncoder(self.symbol_length, input_channel_size, hidden_channel_sizes)

        # change input channel size for decoder(features + symbols + action)
        hidden_channel_sizes[-1] += self.symbol_length + 1
        self.decoder = build_decoder(hidden_channel_sizes)

    def forward(self, state, action):
        features = self.feature_encoder(state) # B, C, H, W
        features_H, features_W = features.shape[2:4]

        symbols = self.symbol_encoder(state) # B, S, 1, 1
        symbols = action.repeat(1, 1, features_H, features_W) # B, S, H, W

        action = convert_action_to_matrix(action, features.shape[2], features.shape[3]) # B, 1, H, W

        decoder_input = torch.cat([features, symbols, action], dim=1) # B, C + S + 1, H, W
        
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