import torch
import torch.nn as nn

from config import *

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size): 
        super(VariationalAutoEncoder, self).__init__()

        self.conv_mu = nn.Conv2d(input_size, latent_size, 4, 1)
        self.conv_log_var = nn.Conv2d(input_size, latent_size, 4, 1)

        self.encoder = nn.Linear(input_size, 400)
        self.mu = nn.Linear(400, latent_size)
        self.logvar = nn.Linear(400, latent_size)
        self.decoder = nn.Linear(latent_size, 400)
        self.output = nn.Linear(400, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = nn.ReLU()(self.encoder(x))
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar)
        x = nn.ReLU()(self.decoder(z))
        x = self.output(x)
        x = x.view(x.size(0), input.size(1), input.size(2), input.size(3))
        return x, mu, logvar

class VCAE(nn.Module): 
    def __init__(self, input_channel_size, hidden_channel_sizes, latent_size): 
        super(VCAE, self).__init__() 

        dimension_reduction = 2 ** len(hidden_channel_sizes)
        if INPUT_SIZE[0] % dimension_reduction != 0: 
            raise ValueError("Input size must be divisible by 2^num_hidden_layers")
        
        encoder_layers = [
            nn.Conv2d(input_channel_size, hidden_channel_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        for i in range(1, len(hidden_channel_sizes)): 
            encoder_layers.append(nn.Conv2d(hidden_channel_sizes[i-1], hidden_channel_sizes[i], 
                                            kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        encoder_output_size = ((INPUT_SIZE[0] // dimension_reduction) ** 2) * hidden_channel_sizes[-1]
        self.variational_part = VariationalAutoEncoder(encoder_output_size, latent_size)

        decoder_layers = []
        for i in range(len(hidden_channel_sizes) - 1, 0, -1):
            decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[i], hidden_channel_sizes[i-1], 
                                                     kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], input_channel_size, kernel_size=2, stride=2))
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.variational_part(x)
        x = self.decoder(x)
        return x, mu, logvar

    def loss_function(self, ground_truth, output):
        input, mu, logvar = output
        reconstruction_loss = nn.MSELoss()(ground_truth, input)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence

if __name__ == "__main__":
    test_input = torch.randn(1, INPUT_SIZE[2], INPUT_SIZE[1], INPUT_SIZE[0])
    model = VCAE(
        input_channel_size=3,
        hidden_channel_sizes=[8, 16, 32],
        latent_size=20
    )
    output = model(test_input)[0]
    print(output.shape)
    assert output.shape == (1, INPUT_SIZE[2], INPUT_SIZE[1], INPUT_SIZE[0])