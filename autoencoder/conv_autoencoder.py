import torch
import torch.nn as nn

from config import INPUT_SIZE

class ConvAutoEncoder(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes): 
        super(ConvAutoEncoder, self).__init__() 

        dimension_reduction = 2 ** len(hidden_channel_sizes)
        if INPUT_SIZE[0] % dimension_reduction != 0: 
            raise ValueError("Input size must be divisible by 2^num_hidden_layers")

        encoder_layers = [
            nn.Conv2d(input_channel_size, hidden_channel_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(True),
        ]
        for i in range(1, len(hidden_channel_sizes)): 
            encoder_layers.append(nn.Conv2d(hidden_channel_sizes[i-1], hidden_channel_sizes[i], 
                                            kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(hidden_channel_sizes) - 1, 0, -1): 
            decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[i], hidden_channel_sizes[i-1], 
                                                     kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], input_channel_size, 
                                                 kernel_size=3, padding=1))
        decoder_layers.append(nn.Tanh())
    
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss_function(self, ground_truth, output):
        return nn.MSELoss()(ground_truth, output)


if __name__ == "__main__":
    # image must have a size as power of 2
    test_input = torch.randn(1, 3, 64, 64)
    model = ConvAutoEncoder(
        input_channel_size=3,
        hidden_channel_sizes=[8, 16, 32]
    )
    for layer in model.encoder:
        test_input = layer(test_input)
        print(test_input.shape)
    for layer in model.decoder:
        test_input = layer(test_input)
        print(test_input.shape)

    assert test_input.shape == (1, 3, 64, 64)