import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_sinusoidal_position_encoding(seq_len, dim, device=None):
    if device is None:
        device = 'cpu'
    positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

# Deformers attention block, with FFT features
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, fine_grained_kernel=10, dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout),
                self.cnn_block(dim, fine_grained_kernel, dropout)
            ]) for _ in range(depth)
        ])

    def cnn_block(self, in_dim, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size, padding=self.get_padding_1D(kernel_size)),
            nn.BatchNorm1d(in_dim),
            nn.ELU()
        )

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))

    def get_fft_feature(self, x):
        fft_magnitude = torch.fft.rfft(x, dim=-1).abs()
        return torch.log(torch.mean(fft_magnitude, dim=-1) + 1e-8)

    def forward(self, x):
        dense_features = []
        for attn, ff, cnn in self.layers:
            x_t = x.transpose(1, 2)
            x_attn = attn(x_t).transpose(1, 2)
            x_cg = x_attn + x
            x_fg = cnn(x)
            fft_feat = self.get_fft_feature(x_fg)
            dense_features.append(fft_feat)
            x_ff = ff(x_cg.transpose(1, 2)).transpose(1, 2)
            x = x_ff + x_fg
        x_pooled = torch.mean(x, dim=-1)
        x_dense = torch.cat(dense_features, dim=-1)
        return torch.cat([x_pooled, x_dense], dim=-1)

class MgTE(nn.Module):
    def __init__(self, num_channels, sampling_rate, embed_dim, num_T, dropout_rate=0.1):
        super().__init__()
        granularities = [0.04, 0.06, 0.08]
        self.kernel_sizes = [int(sampling_rate * g) for g in granularities]
        self.pool_size = 4

        print(self.kernel_sizes)

        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                # CHANGED: Stride is set to kernel_size - 1 and padding is now 0.
                # A max(1, ...) is used to prevent an invalid stride of 0 if a kernel size is 1.
                nn.Conv2d(1, num_T, kernel_size=(1, ks), stride=(1, max(1, ks - 1))),
                nn.LeakyReLU(),
                #nn.BatchNorm2d(num_T)
            ) for ks in self.kernel_sizes
        ])
        self.bn_t = nn.BatchNorm2d(num_T)

        # Shared spatial encoder
        self.spatial_conv = nn.Conv2d(num_T, embed_dim, kernel_size=(num_channels, 1), stride=1)
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(embed_dim)

    def forward(self, x, pool=None):
        if pool is None:
            pool = self.pool_size

        x = x.unsqueeze(1)  # (B, 1, C, T)
        branch_outputs = []

        for branch in self.temporal_branches:
            out = branch(x)
            out = F.max_pool2d(out, kernel_size=(1, pool), stride=(1, pool))
            branch_outputs.append(out)

        #combined = torch.cat(branch_outputs, dim=-1)

        combined = torch.cat(branch_outputs, dim=-1)  # (B, num_T, 1, T_total)
        combined = self.bn_t(combined) 
        s = self.spatial_conv(combined)
        s = self.activation(s)
        s = F.max_pool2d(s, kernel_size=(1, max(1, pool // 4)))
        s = self.bn(s)
        s = s.squeeze(2).permute(0, 2, 1)
        return s

class MGFormer(nn.Module):
    def __init__(self, *, num_chan, num_time, sampling_rate, embed_dim, num_classes, num_T, depth=4, heads=16, mlp_dim=16, dim_head=16, dropout=0.5, fine_grained_kernel=11):
        super().__init__()
        self.token_encoder = MgTE(
            num_channels=num_chan,
            sampling_rate=sampling_rate,
            embed_dim=embed_dim,
            num_T=num_T,
            dropout_rate=dropout
        )

        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fine_grained_kernel=fine_grained_kernel,
            dropout=dropout
        )

        self.mlp_head = nn.Linear(embed_dim * (depth + 1), num_classes)

    def forward(self, x, pool=None):
        tokens = self.token_encoder(x, pool=pool)
        B, L, D = tokens.shape
        pos_emb = get_sinusoidal_position_encoding(L, D, device=x.device)
        tokens = tokens + pos_emb.unsqueeze(0)
        tokens = tokens.permute(0, 2, 1)
        transformer_out = self.transformer(tokens)
        return self.mlp_head(transformer_out)