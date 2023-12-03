import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import *

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
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)

class SlotAttention(torch.nn.Module):
    def __init__(self, n_iters, n_slots, input_dim, slot_dim, mlp_hidden, epsilon=1e-8):
        """
        Adapted to PyTorch from google-research/slot_attention.

        Args:
            n_iters: Number of iterations.
            n_slots: Number of slots.
            input_dim: Dimensionality of input feature vectors.
            slot_dim: Dimensionality of slot feature vectors.
            mlp_hidden: Hidden layer size of MLP.
            epsilon: small value to avoid numerical instability
        """
        super(SlotAttention, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.mlp_hidden = mlp_hidden
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))
        self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)  # applied to slots
        self.to_k = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs
        self.to_v = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP([self.slot_dim, self.mlp_hidden, self.slot_dim])
        
    def forward(self, x):
        n_batch, n_input, n_dim = x.shape
        x = self.norm_inputs(x)
        k = self.to_k(x)  # (n_batch, n_input, slot_dim)
        v = self.to_v(x)  # (n_batch, n_input, slot_dim)

        # initialize slots
        mu = self.slots_mu.expand(n_batch, self.n_slots, self.slot_dim)
        sigma = self.slots_sigma.expand(n_batch, self.n_slots, self.slot_dim)
        slots = mu + sigma * torch.randn_like(mu)  # (n_batch, n_slots, slot_dim)

        # multiple rounds of attention
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # attention
            q = self.to_q(slots)  # (n_batch, n_slots, slot_dim)
            q = q * (self.slot_dim ** -0.5)  # normalize
            attn_logits = k @ q.permute(0, 2, 1)  # (n_batch, n_input, n_slots)
            attn = torch.nn.functional.softmax(attn_logits, dim=-1)

            # weighted mean
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=1, keepdim=True)
            updates = attn.permute(0, 2, 1) @ v  # (n_batch, n_slots, slot_dim)

            # slot update
            slots = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))  # (n_batch * n_slots, slot_dim)
            slots = slots.reshape(n_batch, self.n_slots, self.slot_dim)  # (n_batch, n_slots, slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots

def spatial_broadcast(x, resolution):
    n_batch, n_slots, n_dim = x.shape
    x = x.reshape(n_batch*n_slots, n_dim, 1, 1)
    grid = torch.tile(x, (1, 1, resolution[0], resolution[1]))
    return grid

def build_grid(resolution):
    ranges = [torch.linspace(0, 1, r) for r in resolution]
    grid = torch.stack(torch.meshgrid(*ranges), dim=0)
    grid = torch.cat([grid, 1 - grid], dim=0).unsqueeze(0)
    return grid

class SoftPositionEmbed(torch.nn.Module):
    def __init__(self, hidden_size, resolution):
        super(SoftPositionEmbed, self).__init__()
        self.proj = torch.nn.Linear(4, hidden_size)
        self.grid = build_grid(resolution)
    
    def forward(self, x):
        grid = self.grid.permute(0, 2, 3, 1).to(x.device)
        out = self.proj(grid)
        out = out.permute(0, 3, 1, 2)
        return x + out


class SlotAttentionAutoEncoderV1(nn.Module):
    """
    Slot Attention Auto Encoder V1
    First take the input image and encode it into a feature vector.
    Then use slot attention to get a set of slots.
    Then decode the slots into an image.
    """
    def __init__(self, inchannels, spatial_size, n_slots, n_iters):
        super(SlotAttentionAutoEncoderV1, self).__init__()
        self.spatial_size = spatial_size
        self.n_slots = n_slots
        self.n_iters = n_iters

        self.slot_attention = SlotAttention(
            n_iters=self.n_iters,
            n_slots=self.n_slots,
            input_dim=64,
            slot_dim=64,
            mlp_hidden=128,
        )
        
        n_hidden_channel = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(inchannels, n_hidden_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(n_hidden_channel, n_hidden_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(n_hidden_channel, n_hidden_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(n_hidden_channel, n_hidden_channel, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
        )

        self.decoder_initial_size = (8, 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_hidden_channel, n_hidden_channel, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(n_hidden_channel, n_hidden_channel, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(n_hidden_channel, 4, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
        )

        self.encoder_pos_embed = SoftPositionEmbed(n_hidden_channel, self.spatial_size)
        self.decoder_pos_embed = SoftPositionEmbed(n_hidden_channel, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(n_hidden_channel)
        self.mlp = MLP([n_hidden_channel, n_hidden_channel, n_hidden_channel])

    def forward(self, x):
        n_batch, n_channel, n_height, n_width = x.shape
        x = self.encoder(x) # (n_batch, n_hidden_channel, h, w)
        x = self.encoder_pos_embed(x) # (n_batch, n_hidden_channel, h, w)
        x = x.flatten(2, 3).permute(0, 2, 1) # (n_batch, h * w, n_hidden_channel)
        x = self.mlp(self.layer_norm(x)) # (n_batch, h * w, n_hidden_channel)

        slots = self.slot_attention(x) # (n_batch, n_slots, n_hidden_channel)
        x = spatial_broadcast(slots, self.decoder_initial_size) # (n_batch * n_slots, n_hidden_channel, dec_init_h, dec_init_w)

        x = self.decoder_pos_embed(x) # (n_batch * n_slots, n_hidden_channel, dec_init_h, dec_init_w)
        x = self.decoder(x) # (n_batch * n_slots, 4, h, w)
        x = x.reshape(n_batch, self.n_slots, 4, x.shape[2], x.shape[3]) # (n_batch, n_slots, 4, h, w)
        out, mask = torch.split(x, [3, 1], dim=2)
        # out -> (n_batch, n_slots, 3, h, w)
        # mask -> (n_batch, n_slots, 1, h, w)

        mask = nn.functional.softmax(mask, dim=1)
        result = (out * mask).sum(dim=1)
        return result, out, mask, slots
    
    def loss_function(self, ground_truth, output):
        output = output[0]
        return nn.MSELoss()(ground_truth, output)
    
    def save_other_outputs(self, output, folderpath, filename):
        _, out, mask, slots = output

        out = out.cpu().numpy()
        mask = mask.cpu().numpy()
        slots = slots.cpu().numpy()

        out = out * 0.5 + 0.5
        mask = mask * 0.5 + 0.5
        slots = slots * 0.5 + 0.5

        for i in range(self.n_slots):
            f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
            ax1.imshow(out[0][i].transpose(1, 2, 0))
            ax2.imshow(mask[0][i].transpose(1, 2, 0))
            plt.savefig(f"{folderpath}/{filename}_slot_{i}.png")        
            plt.close()

            # f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
            # ax1.imshow(slots[0][i].transpose(1, 2, 0))
            # ax2.imshow(mask[0][i].transpose(1, 2, 0))
            # plt.savefig(f"{folderpath}/{filename}_slot_{i}_slot.png")        
            # plt.close()





# class SlotAttentionSpatial(torch.nn.Module):
#     def __init__(self, n_iters, n_slots, input_dim, slot_dim, mlp_hidden, epsilon=1e-8):
#         super(SlotAttentionSpatial, self).__init__()
#         self.n_slots = n_slots
#         self.n_iters = n_iters
#         self.input_dim = input_dim
#         self.slot_dim = slot_dim
#         self.mlp_hidden = mlp_hidden
#         self.epsilon = epsilon

#         self.norm_inputs = nn.LayerNorm(self.input_dim)
#         self.norm_slots = nn.LayerNorm(self.slot_dim)
#         self.norm_mlp = nn.LayerNorm(self.slot_dim)

#         self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))
#         self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))

#         self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)  # applied to slots
#         self.to_k = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs
#         self.to_v = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs

#         self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
#         self.mlp = MLP([self.slot_dim, self.mlp_hidden, self.slot_dim])
        
#     def forward(self, x):
#         n_batch, n_input, n_dim = x.shape
#         x = self.norm_inputs(x)
#         k = self.to_k(x)  # (n_batch, n_input, slot_dim)
#         v = self.to_v(x)  # (n_batch, n_input, slot_dim)

#         # initialize slots
#         mu = self.slots_mu.expand(n_batch, self.n_slots, self.slot_dim)
#         sigma = self.slots_sigma.expand(n_batch, self.n_slots, self.slot_dim)
#         slots = mu + sigma * torch.randn_like(mu)  # (n_batch, n_slots, slot_dim)

#         # multiple rounds of attention
#         for _ in range(self.n_iters):
#             slots_prev = slots
#             slots = self.norm_slots(slots)

#             # attention
#             q = self.to_q(slots)  # (n_batch, n_slots, slot_dim)
#             q = q * (self.slot_dim ** -0.5)  # normalize
#             attn_logits = k @ q.permute(0, 2, 1)  # (n_batch, n_input, n_slots)
#             attn = torch.nn.functional.softmax(attn_logits, dim=-1)

#             # weighted mean
#             attn = attn + self.epsilon
#             attn = attn / attn.sum(dim=1, keepdim=True)
#             updates = attn.permute(0, 2, 1) @ v  # (n_batch, n_slots, slot_dim)

#             # slot update
#             slots = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))  # (n_batch * n_slots, slot_dim)
#             slots = slots.reshape(n_batch, self.n_slots, self.slot_dim)  # (n_batch, n_slots, slot_dim)
#             slots = slots + self.mlp(self.norm_mlp(slots))

#         result = torch.einsum('bhs,bhc->bhsc', attn, v)
#         return result

# class SlotAttentionSpatial(torch.nn.Module):
#     def __init__(self, n_iters, n_slots, input_dim, slot_dim, mlp_hidden, epsilon=1e-8):
#         super(SlotAttentionSpatial, self).__init__()
#         self.n_slots = n_slots
#         self.n_iters = n_iters
#         self.input_dim = input_dim
#         self.slot_dim = slot_dim
#         self.mlp_hidden = mlp_hidden
#         self.epsilon = epsilon

#         self.norm_inputs = nn.LayerNorm(self.input_dim)
#         self.norm_slots = nn.LayerNorm(self.slot_dim)
#         self.norm_mlp = nn.LayerNorm(self.slot_dim)

#         self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))
#         self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))

#         self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)  # applied to slots
#         self.to_k = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs
#         self.to_v = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs

#         self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
#         self.mlp = MLP([self.slot_dim, self.mlp_hidden, self.slot_dim])
        
#     def forward(self, x):
#         n_batch, n_input, n_dim = x.shape 
#         x = self.norm_inputs(x)
#         k = self.to_k(x)  # (n_batch, n_input, slot_dim)
#         v = self.to_v(x)  # (n_batch, n_input, slot_dim)

#         # initialize slots
#         mu = self.slots_mu.expand(n_batch, n_input*self.n_slots, self.slot_dim)
#         sigma = self.slots_sigma.expand(n_batch, n_input*self.n_slots, self.slot_dim)
#         slots = mu + sigma * torch.randn_like(mu)  # (n_batch, n_input*n_slots, slot_dim)

#         # multiple rounds of attention
#         for _ in range(self.n_iters):
#             slots_prev = slots
#             slots = self.norm_slots(slots)

#             # attention
#             q = self.to_q(slots)  # (n_batch, n_input*n_slots, slot_dim)
#             q = q * (self.slot_dim ** -0.5)  # normalize
#             attn_logits = k @ q.permute(0, 2, 1)  # (n_batch, n_input, n_input*n_slots)
#             attn = torch.nn.functional.softmax(attn_logits, dim=-1)

#             # weighted mean
#             attn = attn + self.epsilon
#             attn = attn / attn.sum(dim=1, keepdim=True)
#             updates = attn.permute(0, 2, 1) @ v  # (n_batch, n_input*n_slots, slot_dim)

#             # slot update
#             slots = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))  # (n_batch * n_slots * n_input??, slot_dim)
#             slots = slots.reshape(n_batch, n_input*self.n_slots, self.slot_dim)  # (n_batch, n_input*n_slots, slot_dim)
#             slots = slots + self.mlp(self.norm_mlp(slots))

#         slots = slots.reshape(n_batch, n_input, self.n_slots, self.slot_dim)
#         return slots

# class SlotAttentionSpatial(torch.nn.Module):
#     # For this type of slot attention, slot_dim should be equal to input size(eg: 64*64)
#     def __init__(self, n_iters, n_slots, input_dim, slot_dim, mlp_hidden, epsilon=1e-8):
#         super(SlotAttentionSpatial, self).__init__()
#         self.n_slots = n_slots
#         self.n_iters = n_iters
#         self.input_dim = input_dim
#         self.slot_dim = slot_dim
#         self.mlp_hidden = mlp_hidden
#         self.epsilon = epsilon

#         self.norm_inputs = nn.LayerNorm(self.input_dim)
#         self.norm_slots = nn.LayerNorm(self.n_slots)
#         self.norm_mlp = nn.LayerNorm(self.n_slots)

#         self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.n_slots)))
#         self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.n_slots)))

#         self.to_q = nn.Linear(self.n_slots, self.n_slots, bias=False)  # applied to slots
#         self.to_k = nn.Linear(self.input_dim, self.n_slots, bias=False)  # applied to inputs
#         self.to_v = nn.Linear(self.input_dim, self.n_slots, bias=False)  # applied to inputs

#         self.gru = nn.GRUCell(self.n_slots, self.n_slots)
#         self.mlp = MLP([self.n_slots, self.mlp_hidden, self.n_slots])
        
#     def forward(self, x):
#         n_batch, n_input, n_dim = x.shape 
#         x = self.norm_inputs(x)
#         k = self.to_k(x)  # (n_batch, n_input, n_slots)
#         v = self.to_v(x)  # (n_batch, n_input, n_slots)

#         # initialize slots
#         mu = self.slots_mu.expand(n_batch, n_input, self.n_slots)
#         sigma = self.slots_sigma.expand(n_batch, n_input, self.n_slots)
#         slots = mu + sigma * torch.randn_like(mu)  # (n_batch, n_input, n_slots)

#         # multiple rounds of attention
#         for _ in range(self.n_iters):
#             slots_prev = slots
#             slots = self.norm_slots(slots)

#             # attention
#             q = self.to_q(slots)  # (n_batch, n_input, n_slots)
#             q = q * (self.slot_dim ** -0.5)  # normalize
#             attn_logits = k.permute(0, 2, 1) @ q  # (n_batch, n_input, n_input)
#             attn = torch.nn.functional.softmax(attn_logits, dim=-1)

#             # weighted mean
#             attn = attn + self.epsilon
#             attn = attn / attn.sum(dim=1, keepdim=True)
#             updates = v @ attn.permute(0, 2, 1)  # (n_batch, n_input, n_slots)

#             # slot update
#             slots = self.gru(updates.reshape(-1, self.n_slots), slots_prev.reshape(-1, self.n_slots))  # (n_batch * n_input, n_slots)
#             slots = slots.reshape(n_batch, n_input, self.n_slots) # (n_batch, n_input, n_slots)
#             slots = slots + self.mlp(self.norm_mlp(slots))

#         slots = slots.unsqueeze(3) # (n_batch, n_input, n_slots, 1)
#         return slots


class SlotAttentionSpatialv5(torch.nn.Module):
    def __init__(self, n_iters, n_slots, input_dim, slot_dim, mlp_hidden, epsilon=1e-8):
        """
        Adapted to PyTorch from google-research/slot_attention.

        Args:
            n_iters: Number of iterations.
            n_slots: Number of slots.
            input_dim: Dimensionality of input feature vectors.
            slot_dim: Dimensionality of slot feature vectors.
            mlp_hidden: Hidden layer size of MLP.
            epsilon: small value to avoid numerical instability
        """
        super(SlotAttentionSpatialv5, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.mlp_hidden = mlp_hidden
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))
        self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1, 1, self.slot_dim)))

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)  # applied to slots
        self.to_k = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs
        self.to_v = nn.Linear(self.input_dim, self.slot_dim, bias=False)  # applied to inputs

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP([self.slot_dim, self.mlp_hidden, self.slot_dim])
        
    def forward(self, x):
        n_batch, n_input, n_dim = x.shape
        x = self.norm_inputs(x)
        k = self.to_k(x)  # (n_batch, n_input, slot_dim)
        v = self.to_v(x)  # (n_batch, n_input, slot_dim)

        # initialize slots
        mu = self.slots_mu.expand(n_batch, self.n_slots, self.slot_dim)
        sigma = self.slots_sigma.expand(n_batch, self.n_slots, self.slot_dim)
        slots = mu + sigma * torch.randn_like(mu)  # (n_batch, n_slots, slot_dim)

        # multiple rounds of attention
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # attention
            q = self.to_q(slots)  # (n_batch, n_slots, slot_dim)
            q = q * (self.slot_dim ** -0.5)  # normalize
            attn_logits = k @ q.permute(0, 2, 1)  # (n_batch, n_input, n_slots)
            attn = torch.nn.functional.softmax(attn_logits, dim=-1)

            # weighted mean
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=1, keepdim=True)
            updates = attn.permute(0, 2, 1) @ v  # (n_batch, n_slots, slot_dim)

            # slot update
            slots = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))  # (n_batch * n_slots, slot_dim)
            slots = slots.reshape(n_batch, self.n_slots, self.slot_dim)  # (n_batch, n_slots, slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        latest_attn = attn
        return latest_attn

class SlotAttentionAutoEncoderV2(nn.Module):
    """
    Slot Attention Auto Encoder V2
    First take the input image and use slot attention to get a set of slots.
    Then encode the slots into a feature vector.
    Then decode the feature vector into an image.
    """
    def __init__(self, inchannels, hidden_channel_sizes, n_slots, n_iters):
        super(SlotAttentionAutoEncoderV2, self).__init__()
        self.inchannels = inchannels
        self.n_slots = n_slots
        self.n_iters = n_iters

        self.slot_attention = SlotAttentionSpatial(
            n_iters=self.n_iters,
            n_slots=self.n_slots,
            input_dim=3,
            slot_dim=INPUT_SIZE[0] * INPUT_SIZE[1],
            mlp_hidden=128,
        )
        
        encoder_layers = []
        encoder_layers.append(nn.Conv2d(1, hidden_channel_sizes[0], kernel_size=5, stride=1, padding=2))
        encoder_layers.append(nn.ReLU(True))
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
        
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], inchannels + 1, kernel_size=3, padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        self.slot_attention_embed = SoftPositionEmbed(inchannels, INPUT_SIZE[:2])

    def forward(self, x):
        n_batch, n_channel, n_height, n_width = x.shape

        x = self.slot_attention_embed(x) # (n_batch, n_channel, h, w)
        x = x.flatten(2, 3).permute(0, 2, 1) # (n_batch, h * w, n_channel)
        slots = self.slot_attention(x) # (n_batch, h*w, n_slots, n_channel)
        x = slots.reshape(n_batch, n_height, n_width, self.n_slots, 1) # (n_batch, h, w, n_slots, n_channel)
        x = x.permute(0, 3, 4, 1, 2).reshape(n_batch*self.n_slots, 1, n_height, n_width) # (n_batch*n_slots, n_channel, h, w)

        x = self.encoder(x) # (n_batch*n_slots, n_hidden_channel, h, w)

        x = self.decoder(x) # (n_batch*n_slots, n_channel + 1, h, w)
        x = x.reshape(n_batch, self.n_slots, 4, x.shape[2], x.shape[3])

        out, mask = torch.split(x, [self.inchannels, 1], dim=2)
        # out -> (n_batch, n_slots, n_channel, h, w)
        # mask -> (n_batch, n_slots, 1, h, w)

        mask = nn.functional.softmax(mask, dim=1)
        result = (out * mask).sum(dim=1)
        return result, out, mask, slots
    
    def loss_function(self, ground_truth, output):
        output = output[0]
        return nn.MSELoss()(ground_truth, output)
    
    def save_other_outputs(self, output, folderpath, filename):
        _, out, mask, slots = output

        out = out.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        slots = slots.detach().cpu().numpy()

        out = out * 0.5 + 0.5
        mask = mask * 0.5 + 0.5
        slots = slots * 0.5 + 0.5

        for i in range(self.n_slots):
            f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
            ax1.imshow(out[0][i].transpose(1, 2, 0))
            ax2.imshow(mask[0][i].transpose(1, 2, 0))
            plt.savefig(f"{folderpath}/{filename}_slot_{i}.png")        
            plt.close()


class SlotAttentionAutoEncoderv5(nn.Module): 
    """
    Slot Attention Auto Encoder V5
    """
    def __init__(self, inchannels, hidden_channel_sizes, n_slots, n_iters):
        super(SlotAttentionAutoEncoderv5, self).__init__()
        self.inchannels = inchannels
        self.n_slots = n_slots
        self.n_iters = n_iters

        self.slot_attention = SlotAttentionSpatialv5(
            n_iters=self.n_iters,
            n_slots=self.n_slots,
            input_dim=3,
            slot_dim=64,
            mlp_hidden=128,
        )
        
        encoder_layers = []
        encoder_layers.append(nn.Conv2d(3, hidden_channel_sizes[0], kernel_size=5, stride=1, padding=2))
        encoder_layers.append(nn.ReLU(True))
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
        
        decoder_layers.append(nn.ConvTranspose2d(hidden_channel_sizes[0], inchannels, kernel_size=3, padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        self.slot_attention_embed = SoftPositionEmbed(inchannels, INPUT_SIZE[:2])

    def forward(self, initial_input):
        n_batch, n_channel, n_height, n_width = initial_input.shape

        x = self.slot_attention_embed(initial_input) # (n_batch, n_channel, h, w)
        x = x.flatten(2, 3).permute(0, 2, 1) # (n_batch, h * w, n_channel)
        slots = self.slot_attention(x) # (n_batch, h*w, n_slots)
        slots = nn.functional.softmax(slots, dim=1)
        slots = slots.reshape(n_batch, n_height, n_width, self.n_slots).permute(0, 3, 1, 2).unsqueeze(2) # (n_batch, n_slots, 1, h, w)
        x = slots * initial_input.unsqueeze(1) # (n_batch, n_slots, n_channel, h, w)
        # slots = torch.einsum('bnhw,bchw->bnchw', slots, initial_input) # (n_batch, n_slots, n_channel, h, w)
        x = x.reshape(n_batch*self.n_slots, n_channel, n_height, n_width) # (n_batch*n_slots, n_channel, h, w)

        x = self.encoder(x) # (n_batch*n_slots, n_hidden_channel, h, w)

        x = self.decoder(x) # (n_batch*n_slots, n_channel, h, w)
        out = x.reshape(n_batch, self.n_slots, n_channel, x.shape[2], x.shape[3]) # (n_batch, n_slots, n_channel, h, w)

        result = out.sum(dim=1)
        return result, out, slots
    
    def loss_function(self, ground_truth, output):
        output = output[0]
        return nn.MSELoss()(ground_truth, output)
    
    def save_other_outputs(self, output, folderpath, filename):
        _, out, slots = output

        out = out.detach().cpu().numpy()
        slots = slots.detach().cpu().numpy()
        out = out * 0.5 + 0.5
        slots = slots * 0.5 + 0.5

        for i in range(self.n_slots):
            f, [ax1, ax2] = plt.subplots(1, 2, figsize=(32, 10))
            ax1.imshow(out[0][i].transpose(1, 2, 0))
            ax2.imshow(slots[0][i].transpose(1, 2, 0))
            plt.savefig(f"{folderpath}/{filename}_slot_{i}.png")        
            plt.close()
