import torch
import torch.nn as nn

class LinearAutoEncoder(nn.Module): 
    def __init__(self, input_size, hidden_channel_sizes): 
        super(LinearAutoEncoder, self).__init__() 

        encoder_layers = [
            nn.Linear(input_size, hidden_channel_sizes[0]),
            nn.ReLU(True)
        ]
        for i in range(1, len(hidden_channel_sizes)): 
            encoder_layers.append(nn.Linear(hidden_channel_sizes[i-1], hidden_channel_sizes[i]))
            encoder_layers.append(nn.ReLU(True))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(hidden_channel_sizes) - 1, 0, -1): 
            decoder_layers.append(nn.Linear(hidden_channel_sizes[i], hidden_channel_sizes[i-1]))
            decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.Linear(hidden_channel_sizes[0], input_size))
        decoder_layers.append(nn.Tanh())
    
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == "__main__":
    test_input = torch.randn(1, 28*28)
    model = LinearAutoEncoder(
        input_size=28*28,
        hidden_channel_sizes=[128, 64, 32, 8]
    )
    for layer in model.encoder:
        test_input = layer(test_input)
        print(test_input.shape)
    for layer in model.decoder:
        test_input = layer(test_input)
        print(test_input.shape)
    assert test_input.shape == (1, 28*28)
