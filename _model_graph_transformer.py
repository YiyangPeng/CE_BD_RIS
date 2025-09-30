import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

List = nn.ModuleList

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual

class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb=None,
        dim_head=64,
        heads=8,
        edge_dim=1
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.pos_emb = None

    def forward(self, nodes, edges, mask=None):
        h = self.heads
        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)
        e_kv = self.edges_to_kv(edges)
        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))
        if exists(self.pos_emb):
            pass

        ek, ev = e_kv, e_kv
        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward

def FeedForward(dim, ff_mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )

# classes

class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=1,
        heads=8,
        with_feedforwards=False,
        norm_edges=False,
        accept_adjacency_matrix=False
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()
        self.adj_emb = nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None
        for _ in range(depth):
            self.layers.append(List([
                List([
                    PreNorm(dim, Attention(dim, pos_emb=None, edge_dim=edge_dim, dim_head=dim_head, heads=heads)),
                    GatedResidual(dim)
                ]),
                List([
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))

    def forward(
        self,
        nodes,
        edges=None, # edge attr. (batch_size, num_nodes, num_nodes, edge_dim)
        adj_mat=None,
        mask=None
    ):
        batch, seq, _ = nodes.shape
        if exists(edges):
            edges = self.norm_edges(edges)
        all_edges = default(edges, 0) + default(adj_mat, 0)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, all_edges, mask = mask), nodes)
            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)
        return nodes


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
        # Encorders
        self.node_feat_init = FeedForward2(input_dim, hidden_dim, hidden_dim)
        # Decorders
        self.decoder_channel_est = FeedForward2(
            hidden_dim, hidden_dim, 2 * self.n_antennas_bs * self.n_antennas_EUs * (self.n_elements_per_block**2) * self.n_blocks)
        # Graph Trans.
        self.model = GraphTransformer(dim=hidden_dim, depth=config.model.n_layers) # Attention Mech
        #self.model = FeedForward2(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, x, edge_feat):
        '''
        :param x: node feats. edge_feat: distance between EUs.
        (feats. should be vectorized form with real and imaginary components separated)
        :return:
        '''
        B = x.size(0)
        # Initialize node feats. (feats. should be vectorized form with real and imaginary components separated)
        x = self.node_feat_init(x) # 2 layer mlp, encoder
        # Graph Trans.
        x = self.model(x, edge_feat)
        # DECODING
        # Channel Est.
        Q = self.decoder_channel_est(x).view(
            B, self.n_EUs, 2, self.n_antennas_bs * self.n_antennas_EUs, self.n_elements_per_block**2 * self.n_blocks) # 2 layer mlp
        return Q

def FeedForward2(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim)
    )


