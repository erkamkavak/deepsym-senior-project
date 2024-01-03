import torch
import torch.nn as nn

from config import INPUT_SIZE
from .model_utils import Encoder, Decoder, MLP

class ConvPredictor(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes, 
                 pretrained_model_name=None, 
                 is_encoder_pretrained=False, 
                 is_decoder_pretrained=False): 
        super(ConvPredictor, self).__init__() 

        if is_encoder_pretrained: 
            self.feature_encoder = Encoder.load_model(input_channel_size, hidden_channel_sizes, pretrained_model_name)
        else: 
            self.feature_encoder = Encoder(input_channel_size, hidden_channel_sizes)

        mlp_layers = [
            hidden_channel_sizes[-1] + 1, 
            hidden_channel_sizes[-1], 
            hidden_channel_sizes[-1],
        ]
        self.next_state_feature_predictor = MLP(mlp_layers)
        # self.next_state_feature_predictor = nn.MultiheadAttention(embed_dim=, num_heads=)

        if is_decoder_pretrained: 
            self.feature_decoder = Decoder.load_model(hidden_channel_sizes, input_channel_size, pretrained_model_name)
        else: 
            self.feature_decoder = Decoder(hidden_channel_sizes, input_channel_size)

    def forward(self, state, action):
        features = self.feature_encoder(state) # B, C, H, W

        # action -> (B) -> (B, 1, 1, 1) -> (B, 1, H, W)
        action = action.unsqueeze(1).unsqueeze(2).unsqueeze(3) # B, 1, 1, 1
        action = action.repeat(1, features.shape[2], features.shape[3], 1) # B, H, W, 1

        features = features.permute(0, 2, 3, 1) # B, H, W, C

        # B, C + 1, H, W -> B, C, H, W
        next_state_features = self.next_state_feature_predictor(torch.cat((features, action), dim=3))
        next_state_features = next_state_features.permute(0, 3, 1, 2) # B, C, H, W

        next_state = self.feature_decoder(next_state_features)
        return next_state, next_state_features

    def loss_function(self, ground_truth, output):
        # next_state_pred, next_state_features_pred = output
        # next_state_features = self.feature_encoder(ground_truth)
        # feature_loss = nn.MSELoss()(next_state_features, next_state_features_pred)

        return nn.MSELoss()(ground_truth, output[0])
    
    def save_encoder(self, save_path): 
        self.feature_encoder.save_model(save_path)
        self.feature_decoder.save_model(save_path)

    def save_other_outputs(self, output, path, name): 
        pass

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