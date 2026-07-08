import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils.train_helpers import weight_init
from utils.attention_layers import flash_attention
import math
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Taken from https://github.com/facebookresearch/DiT
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    Taken from https://github.com/facebookresearch/DiT
    grid_size_h, grid_size_w: height/width of the grid
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim] or [1+grid_size_h*grid_size_w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def patchify_depth(depth, patch_size=16):
    """
    depth: [B, H, W] or [B,1,H,W]
    returns patches: [B, num_patches, 1], the mean depth per patch.
    """
    if depth.dim() == 4:
        # [B, 1, H, W]
        depth = depth.squeeze(1)
    B, H, W = depth.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Make sure H and W divisible by patch_size"
    ph = H // patch_size
    pw = W // patch_size
    # reshape to patches
    depth = depth.view(B, ph, patch_size, pw, patch_size)  # [B, ph, p, pw, p]
    depth = depth.permute(0, 1, 3, 2, 4).contiguous()      # [B, ph, pw, p, p]
    depth = depth.view(B, ph * pw, patch_size * patch_size)  # [B, num_patches, patch_area]
    # pooled scalar per patch:
    depth_patch_mean = depth.mean(dim=-1, keepdim=True)  # [B, num_patches, 1]
    return depth_patch_mean  # [B, num_patches, 1]


class AttentionLayerDiT(MessagePassing):
    """Transformer attention layer for DiT, taken from https://github.com/facebookresearch/DiT"""
    def __init__(self,
                 hidden_dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(AttentionLayerDiT, self).__init__(aggr='add', node_dim=0, **kwargs)
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_dim, self.head_dim * num_heads, bias=qkv_bias)
        
        # Optional normalization for q and k
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        # Attention and projection dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def message(self, q_i, k_j, v_j, index, ptr):
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs):
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        return inputs

    def _attn_block(self, x_src, x_dst, edge_index):
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        x_dst =  self.propagate(edge_index=edge_index, q=q, k=k, v=v)

        x_dst = self.proj(x_dst)
        x_dst = self.proj_drop(x_dst)
        return x_dst
    
    def forward(self, x, edge_index):
        x_src = x_dst = x
        return self._attn_block(x_src, x_dst, edge_index)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    Taken from https://github.com/facebookresearch/DiT
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Taken from https://github.com/facebookresearch/DiT
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        
        embeddings = self.embedding_table(labels)
        return embeddings


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Taken from https://github.com/facebookresearch/DiT
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class CoarseToFineCrossAttention(nn.Module):
    """
    Cross-attention that combines a full attention path with a gated Perceiver
    bottleneck path attending to a small set of compressed latent tokens.

    Two parallel paths process the same Q/KV:
      Path A (full):   Standard flash cross-attention against the full kv sequence.
      Path B (coarse): Perceiver bottleneck (latents compress kv, then q reads
                        from the compressed latents).
    The outputs are combined as:
      out = out_full + tanh(gate) * out_coarse
    where gate is a learnable scalar initialized to 0, so the module starts as
    an exact copy of the full-attention path and can never perform worse.

    Shared K/V projections keep parameter overhead small.

    q:     scenario tokens  [N_q, D]   (flattened across batch)
    kv:    image/text tokens [B, Nk, D]
    batch: scene id per query token [N_q]
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.0, qk_norm=True,
                 num_latents=32):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Shared Q/K/V projections (used by full path + coarse stage 1 K/V)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.q_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()

        # Coarse path — stage 1: latents attend to KV
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        self.lat_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lat_q_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()
        self.lat_norm = nn.LayerNorm(hidden_dim)

        # Coarse path — stage 2: Q attends to compressed latents
        self.lat_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lat_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lat_k_norm = nn.LayerNorm(hidden_dim) if qk_norm else nn.Identity()
        self.lat_out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Flamingo-style gate (init 0 → starts as full-attention baseline)
        self.gate = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, kv, batch):
        device = q.device
        B, Nk, D = kv.shape
        H, Dh = self.num_heads, self.head_dim

        # ---- pack q into [B, Lq, D] ----
        idx_lists = [(batch == b).nonzero(as_tuple=True)[0] for b in range(B)]
        q_lens = torch.tensor([len(i) for i in idx_lists], device=device, dtype=torch.long)
        Lq = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
        Q_pad = q.new_zeros((B, Lq, D))
        for b in range(B):
            n_b = int(q_lens[b].item())
            if n_b > 0:
                Q_pad[b, :n_b] = q[idx_lists[b]]

        # ---- shared projections ----
        Q = self.q_norm(self.q_proj(Q_pad)).view(B, Lq, H, Dh)
        K = self.k_norm(self.k_proj(kv)).view(B, Nk, H, Dh)
        V = self.v_proj(kv).view(B, Nk, H, Dh)
        k_lens = torch.tensor([Nk] * B, device=device)

        # ===== Path A: full flash attention (baseline) =====
        out_full = flash_attention(
            q=Q, k=K, v=V,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
            unflatten=False,
        )                                                       # [N_q, H, Dh]
        out_full = self.out_proj(out_full.flatten(1))           # [N_q, D]

        # ===== Path B: Perceiver bottleneck =====
        L = self.latent_tokens.expand(B, -1, -1)               # [B, Nl, D]
        Nl = L.shape[1]

        # Stage 1: latents attend to KV (reuse shared K, V)
        LQ = self.lat_q_norm(self.lat_q_proj(L)).view(B, Nl, H, Dh)
        latent_out = flash_attention(
            q=LQ, k=K, v=V,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
        )                                                       # [B, Nl, H, Dh]
        L = self.lat_norm(L + latent_out.reshape(B, Nl, D))    # residual + norm

        # Stage 2: Q attends to compressed latents
        LK2 = self.lat_k_norm(self.lat_k_proj(L)).view(B, Nl, H, Dh)
        LV2 = self.lat_v_proj(L).view(B, Nl, H, Dh)
        lat_k_lens = torch.tensor([Nl] * B, device=device)

        out_coarse = flash_attention(
            q=Q, k=LK2, v=LV2,
            q_lens=q_lens, k_lens=lat_k_lens,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
            unflatten=False,
        )                                                       # [N_q, H, Dh]
        out_coarse = self.lat_out_proj(out_coarse.flatten(1))   # [N_q, D]

        # ===== combine with gated residual =====
        out = out_full + torch.tanh(self.gate) * out_coarse
        out = self.proj_drop(out)
        return out


CROSSATTENTION_CLASSES = {
    'img_crossglobal': CoarseToFineCrossAttention,
    'text_crossglobal': CoarseToFineCrossAttention,
    None: None,
}


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Optionally cross-attends to a conditioning context (image or text tokens).
    Taken from https://github.com/facebookresearch/DiT
    """
    def __init__(self, hidden_size, num_heads, cross_attn_type, dropout=0.0, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionLayerDiT(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        cross_attn_class = CROSSATTENTION_CLASSES.get(cross_attn_type, None)
        self.cross_attn = cross_attn_class(hidden_size, num_heads, dropout) if cross_attn_class is not None else None
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_cross = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        ) if self.cross_attn is not None else None

    def forward(self, x, c, edge_index, cond=None, batch=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), edge_index)
        if self.cross_attn is not None and cond is not None:
            shift_cross, scale_cross, gate_cross = self.adaLN_modulation_cross(c).chunk(3, dim=1)
            x = x + gate_cross * self.cross_attn(modulate(self.norm2(x), shift_cross, scale_cross), cond, batch)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FactorizedDiTBlockCond(nn.Module):
    """
    Sequence of factorized (a2l, l2l, l2a, a2a) DiT blocks, optionally cross-attending
    to a conditioning context (image or text tokens) at every attention stage.
    """
    def __init__(
            self,
            hidden_dim,
            hidden_dim_agent,
            num_heads,
            num_heads_agent,
            dropout,
            mlp_ratio=4.0,
            num_l2l_blocks=1,
            cross_attn_type=None):

        super().__init__()
        self.num_l2l_blocks = num_l2l_blocks
        self.cross_attn_type = cross_attn_type

        # l2l
        # we stack several l2l blocks to give more capacity for lane modeling
        self.l2l_blocks = []
        for _ in range(num_l2l_blocks):
            self.l2l_blocks.append(DiTBlock(hidden_dim, num_heads, cross_attn_type, dropout, mlp_ratio))
        self.l2l_blocks = nn.ModuleList(self.l2l_blocks)

        #a2a
        self.a2a_block = DiTBlock(hidden_dim_agent, num_heads_agent, cross_attn_type, dropout, mlp_ratio)

        # l2a
        self.downsample_x_lane = nn.Linear(hidden_dim, hidden_dim_agent)
        self.l2a_block = DiTBlock(hidden_dim_agent, num_heads_agent, cross_attn_type, dropout, mlp_ratio)

        # a2l
        self.upsample_x_agent = nn.Linear(hidden_dim_agent, hidden_dim)
        self.a2l_block = DiTBlock(hidden_dim, num_heads, cross_attn_type, dropout, mlp_ratio)


    def forward(
            self,
            x_lane,
            x_agent,
            c,
            c_small,
            l2l_edge_index,
            a2a_edge_index,
            l2a_edge_index,
            text_context=None,
            img_lane_context=None,
            img_agent_context=None,
            lane_batch=None,
            agent_batch=None):

        if self.cross_attn_type is None:
            cond_context_lanes = None
            cond_context_agents = None
        elif self.cross_attn_type.startswith("text") and text_context is not None:
            cond_context_lanes, cond_context_agents = text_context
        elif self.cross_attn_type.startswith("img") and img_lane_context is not None and img_agent_context is not None:
            cond_context_lanes = img_lane_context
            cond_context_agents = img_agent_context
        else:
            raise ValueError(f"Unsupported cross-attention type: {self.cross_attn_type}")

        # a2l
        x_lane_agent = torch.cat([x_lane, self.upsample_x_agent(x_agent)], dim=0)
        lane_agent_batch = torch.cat([lane_batch, agent_batch], dim=0)
        x_lane_agent = self.a2l_block(x_lane_agent, c, l2a_edge_index[[1, 0], :], cond_context_lanes, lane_agent_batch)
        x_lane = x_lane_agent[:x_lane.shape[0]]

        # l2l
        for i in range(self.num_l2l_blocks):
            x_lane = self.l2l_blocks[i](x_lane, c[:x_lane.shape[0]], l2l_edge_index, cond_context_lanes, lane_batch)

        # l2a
        x_lane_agent = torch.cat([self.downsample_x_lane(x_lane), x_agent], dim=0)
        x_lane_agent = self.l2a_block(x_lane_agent, c_small, l2a_edge_index, cond_context_agents, lane_agent_batch)
        x_agent = x_lane_agent[x_lane.shape[0]:]

        # a2a
        x_agent = self.a2a_block(x_agent, c_small[x_lane.shape[0]:], a2a_edge_index, cond_context_agents, agent_batch)

        return x_lane, x_agent


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    Taken from https://github.com/facebookresearch/DiT
    """
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, latent_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TwoLayerResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerResMLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.transform_linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.transform_norm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        
        x = self.transform_linear(x)
        x = self.transform_norm(x)
        
        out  = out + x
        out = self.relu(out)
        return out