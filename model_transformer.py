import torch
import torch.nn as nn

class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        '''
        → MultiHeadAttention → (Dropout → +Residual → LayerNorm)
        → FeedForward → (...) → Output
        '''
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff_output))
        return src


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        for layer in self.layers:
            x = layer(x)
        return x


def FeedForward_2Layer(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


class Net(nn.Module):
    """ Default Model - GCN
        Model: x→GCNconv→BN→ReLU→Dropout→x . . .x→GCNConv→out
    """
    def __init__(self, config):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_layers = config.model.n_layers
        input_dim = 2 * config.system.pilot_length * config.system.n_antennas_bs * config.system.n_antennas_EUs
        hidden_dim = config.model.hidden_dim
        # System Settings
        self.n_elements_per_block = config.system.ris.n_elements_per_block
        self.n_blocks = config.system.ris.n_blocks
        self.n_antennas_bs = config.system.n_antennas_bs
        self.n_antennas_EUs = config.system.n_antennas_EUs
        self.n_upper_tri_elements = int(self.n_elements_per_block*(self.n_elements_per_block + 1) / 2)
        self.n_EUs = config.system.n_EUs
        # Encoder:
        self.node_feat_init = FeedForward_2Layer(input_dim, hidden_dim, hidden_dim)
        # Decorder:
        self.decoder_channel_est = FeedForward_2Layer(
            hidden_dim, hidden_dim, 2 * self.n_antennas_bs * self.n_antennas_EUs * (self.n_elements_per_block**2) * self.n_blocks)
        # Graph Trans.
        self.model = SimpleTransformer(
            d_model=config.model.hidden_dim, nhead=4, num_layers=config.model.n_layers, dim_feedforward=config.model.hidden_dim * 2, dropout=0.0)

    def forward(self, x):
        '''
        :param x: node feats. edge_feat: distance between EUs.
        (feats. should be vectorized form with real and imaginary components separated)
        :return:
        '''
        B = x.size(0)
        # Initialize node feats. (feats. should be vectorized form with real and imaginary components separated)
        x = self.node_feat_init(x) # 2 layer mlp, encoder
        # Graph Trans.
        x = self.model(x)
        # DECODING
        # Channel Est.
        Q = self.decoder_channel_est(x).view(
            B, self.n_EUs, 2, self.n_antennas_bs * self.n_antennas_EUs, self.n_elements_per_block**2 * self.n_blocks) # 2 layer mlp
        return Q
