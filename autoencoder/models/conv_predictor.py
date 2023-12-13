import torch
import torch.nn as nn

from config import INPUT_SIZE
from .model_utils import Encoder, build_decoder

class ConvPredictor(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes): 
        super(ConvPredictor, self).__init__() 

        self.feature_encoder = Encoder(input_channel_size, hidden_channel_sizes)
        hidden_channel_sizes[-1] += 1 # add action channel
        self.decoder = build_decoder(hidden_channel_sizes, input_channel_size) 

    def forward(self, state, action):
        features = self.feature_encoder(state) # B, C, H, W

        action = action.unsqueeze(2).unsqueeze(3) # B, 1, 1, 1
        action = action.repeat(1, 1, features.shape[2], features.shape[3]) # B, 1, H, W
        features = torch.cat((features, action), dim=1) # B, C + 1, H, W
        next_state = self.decoder(features)
        return next_state

    def loss_function(self, ground_truth, output):
        return nn.MSELoss()(ground_truth, output)
    
    def save_encoder(self, save_path): 
        self.feature_encoder.save_model(save_path)


if __name__ == "__main__":
    # image must have a size as power of 2
    test_input = torch.randn(1, 3, 64, 64)
    model = ConvPredictor(
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