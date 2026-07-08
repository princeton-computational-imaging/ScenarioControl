import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils.dit_layers import FactorizedDiTBlockCond, FinalLayer, LabelEmbedder, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, TwoLayerResMLP, patchify_depth
from utils.pyg_helpers import get_indices_within_scene


def cond_mapping(cfg):
    """Maps the model's conditioning flags to a CROSSATTENTION_CLASSES key."""
    cond_type = ""
    if cfg.img_conditioning:
        cond_type += "img_crossglobal"
    if cfg.text_conditioning:
        cond_type += "text_crossglobal"
    return cond_type if cond_type != "" else None


class DiT(nn.Module):

    def __init__(self, cfg):
        super(DiT, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        self.conditioning_type = cond_mapping(self.cfg_model)

        self.emb_drop = nn.Dropout(self.cfg_model.dropout)

        # Condition on scene type
        self.scene_type_embedder = LabelEmbedder(self.cfg_dataset.num_map_ids * self.cfg_model.num_scene_types, self.cfg_model.hidden_dim, self.cfg_model.label_dropout)

        # Condition on number of agents and lanes
        self.num_agents_embedder = LabelEmbedder(self.cfg_dataset.max_num_agents + 1, self.cfg_model.hidden_dim, 0)
        self.num_lanes_embedder = LabelEmbedder(self.cfg_dataset.max_num_lanes + 1, self.cfg_model.hidden_dim, 0)

        # Diffusion timestep embedding
        self.t_embedder = TimestepEmbedder(self.cfg_model.hidden_dim)
        # Used because agent embedding is smaller than lane embedding
        self.downsample_c = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)

        # Embed agent and lane latents
        self.lane_embedder = TwoLayerResMLP(self.cfg_model.lane_latent_dim, self.cfg_model.hidden_dim)
        self.agent_embedder = TwoLayerResMLP(self.cfg_model.agent_latent_dim, self.cfg_model.agent_hidden_dim)

        # These will be overwritten by sin/cos positional encodings
        self.pos_emb_lane = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_lanes, self.cfg_model.hidden_dim), requires_grad=False)
        self.pos_emb_agent = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_agents, self.cfg_model.agent_hidden_dim), requires_grad=False)

        if self.cfg_model.text_conditioning:
            # text embedding projection
            self.text_embedding_lanes = nn.Sequential(
                nn.Linear(self.cfg_model.text_feat_dim, self.cfg_model.hidden_dim), nn.GELU(approximate='tanh'),
                nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.hidden_dim))
            self.text_embedding_agents = nn.Sequential(
                nn.Linear(self.cfg_model.text_feat_dim, self.cfg_model.agent_hidden_dim), nn.GELU(approximate='tanh'),
                nn.Linear(self.cfg_model.agent_hidden_dim, self.cfg_model.agent_hidden_dim))

        # image conditioning
        if self.cfg_model.img_conditioning:
            # Project DINO patch features (C -> img_token_dim)
            self.dino_proj = nn.Linear(self.cfg_model.dino_feat_channels, self.cfg_model.hidden_dim)

            # Project depth patches (1 -> img_token_dim)
            self.depth_proj = nn.Linear(1, self.cfg_model.hidden_dim)

            # Separate tiny projections so agent/agent-small cross-attn can use appropriate dims
            self.img_to_agent_proj = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)
            self.img_to_lane_proj = nn.Identity()  # already same dim as lane hidden_dim

            # Precompute max positional embedding for shared image space
            PH, PW = self.cfg_model.dino_dim_h, self.cfg_model.dino_dim_w
            self.pos_emb_img = nn.Parameter(
                torch.zeros(PH * PW, self.cfg_model.hidden_dim), requires_grad=False
            )

        # factorized dit blocks
        self.blocks = nn.ModuleList([
            FactorizedDiTBlockCond(
                self.cfg_model.hidden_dim,
                self.cfg_model.agent_hidden_dim,
                self.cfg_model.num_heads,
                self.cfg_model.agent_num_heads,
                self.cfg_model.dropout,
                mlp_ratio=4,
                num_l2l_blocks=self.cfg_model.num_l2l_blocks,
                cross_attn_type=self.conditioning_type,
                ) for _ in range(self.cfg_model.num_factorized_dit_blocks)
        ])

        # noise prediction heads
        self.pred_agent_noise = FinalLayer(self.cfg_model.agent_hidden_dim, self.cfg_model.agent_latent_dim)
        self.pred_lane_noise = FinalLayer(self.cfg_model.hidden_dim, self.cfg_model.lane_latent_dim)
        self.initialize_weights()


    def initialize_weights(self):
        """ Custom initialization for DiT model"""
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) lane and agent pos_embed by sin-cos embedding:
        pos_emb_lane = get_1d_sincos_pos_embed_from_grid(self.pos_emb_lane.shape[-1], np.arange(self.pos_emb_lane.shape[0]))
        self.pos_emb_lane.data.copy_(torch.from_numpy(pos_emb_lane).float())
        pos_emb_agent = get_1d_sincos_pos_embed_from_grid(self.pos_emb_agent.shape[-1], self.cfg_dataset.max_num_lanes + np.arange(self.pos_emb_agent.shape[0]))
        self.pos_emb_agent.data.copy_(torch.from_numpy(pos_emb_agent).float())

        if self.cfg_model.text_conditioning:
            # Initialize text embedding projection
            nn.init.normal_(self.text_embedding_lanes[0].weight, std=0.02)
            nn.init.normal_(self.text_embedding_lanes[2].weight, std=0.02)
            nn.init.normal_(self.text_embedding_agents[0].weight, std=0.02)
            nn.init.normal_(self.text_embedding_agents[2].weight, std=0.02)

        if self.cfg_model.img_conditioning:
            pos_img = get_2d_sincos_pos_embed(self.cfg_model.hidden_dim, self.cfg_model.dino_dim_h, self.cfg_model.dino_dim_w)
            self.pos_emb_img.data.copy_(torch.from_numpy(pos_img).float())

        # Initialize label embedding table:
        nn.init.normal_(self.scene_type_embedder.embedding_table.weight, std=0.02)

        # Initialize num lane and num agent embedding tables:
        nn.init.normal_(self.num_agents_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.num_lanes_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            for l2l_block in block.l2l_blocks:
                nn.init.constant_(l2l_block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(l2l_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2l_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_agent_noise.linear.weight, 0)
        nn.init.constant_(self.pred_agent_noise.linear.bias, 0)

        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_lane_noise.linear.weight, 0)
        nn.init.constant_(self.pred_lane_noise.linear.bias, 0)


    def _prepare_image_tokens(self, dino_feats, depth_map):
        """
        dino_feats: [B, C, PH, PW]  (ViT patch tokens, e.g. 14x14)
        depth_map: [B, 1, H, W]     (absolute depth, resized to multiple of patch_size)

        Returns:
            img_tokens_lane:  [B, N_img, hidden_dim]        (for lane cross-attn)
            img_tokens_agent: [B, N_img, agent_hidden_dim]  (for agent cross-attn)
        """
        B = dino_feats.shape[0]

        # ---- Shared Positional Embedding ----
        pos = self.pos_emb_img.unsqueeze(0).expand(B, -1, -1)

        # DINO tokens
        dino_flat = dino_feats.flatten(2).permute(0, 2, 1).contiguous()  # [B, PH*PW, C]
        dino_tokens = self.dino_proj(dino_flat)                          # [B, PH*PW, hidden_dim]
        dino_tokens = dino_tokens + pos

        # Depth tokens (patchify first, using the same patch size as the DINO grid so
        # depth and DINO tokens line up and can share the same positional embedding)
        depth_patches = patchify_depth(depth_map, patch_size=self.cfg_model.dino_patch_size)  # [B, num_patches, 1]
        depth_tokens = self.depth_proj(depth_patches)  # [B, num_patches, hidden_dim]
        depth_tokens = depth_tokens + pos

        # Concatenate + project
        img_tokens = torch.cat([dino_tokens, depth_tokens], dim=1)  # [B, N_img, hidden_dim]
        img_tokens_lane = self.img_to_lane_proj(img_tokens)    # [B, N_img, hidden_dim]
        img_tokens_agent = self.img_to_agent_proj(img_tokens)  # [B, N_img, agent_hidden_dim]

        return img_tokens_lane, img_tokens_agent

    def _sample_cond_drop_mask(self, num_scenes: int, device, unconditional: bool, drop_prob: float = 0.0):
        """
        Returns a boolean mask [num_scenes] where:
        True  = drop conditioning
        False = keep conditioning

        If `unconditional` is True, drop all conditions → mask = all True.
        """
        if unconditional:
            return torch.ones(num_scenes, dtype=torch.bool, device=device)
        else:
            # drop with probability = drop_prob
            drop_mask = torch.rand(num_scenes, device=device) < drop_prob
            return drop_mask

    def forward(self,
                x_agent,
                x_lane,
                data,
                agent_timestep,
                lane_timestep,
                text_embeds=None,
                unconditional=False,
                drop=False,
                drop_prob_text=0.3
                ):
        """ Forward pass of the DiT model."""
        lane_idx_batch = get_indices_within_scene(data['lane'].batch)
        agent_idx_batch = get_indices_within_scene(data['agent'].batch)

        # add positional embeddings
        pos_emb_lane = self.pos_emb_lane[lane_idx_batch]
        pos_emb_agent = self.pos_emb_agent[agent_idx_batch]

        x_lane = self.lane_embedder(x_lane[:, 0]) + pos_emb_lane
        x_agent = self.agent_embedder(x_agent[:, 0]) + pos_emb_agent

        scene_idx = self.cfg_dataset.num_map_ids * data['lg_type'].long() + data['map_id'].long()
        num_scenes = scene_idx.shape[0]

        # sample one drop mask per scene (True = drop all cond)
        drop_prob = drop_prob_text if drop else 0.0  # the drop should not happen at test time
        drop_mask = self._sample_cond_drop_mask(num_scenes, scene_idx.device, unconditional, drop_prob)

        scene_type = self.scene_type_embedder(
            scene_idx.long(),
            train=self.training,
            force_drop_ids=torch.ones_like(scene_idx) if unconditional else None
        )

        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch
        agent_scene_type = scene_type[agent_batch]
        lane_scene_type = scene_type[lane_batch]

        num_agents = data['num_agents'].long()
        num_lanes = data['num_lanes'].long()
        num_agents_emb = self.num_agents_embedder(num_agents, train=self.training)[agent_batch]
        num_lanes_emb = self.num_lanes_embedder(num_lanes, train=self.training)[lane_batch]

        # embedding of timestep
        t = self.t_embedder(torch.cat([lane_timestep, agent_timestep], dim=-1))
        # embedding of number of agents and lanes
        n = torch.cat([num_lanes_emb, num_agents_emb], dim=0)
        # embedding of scene type
        y = torch.cat([lane_scene_type, agent_scene_type], dim=0)

        l2l_edge_index = data['lane', 'to', 'lane'].edge_index
        a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        l2a_edge_index = data['lane', 'to', 'agent'].edge_index.clone()
        l2a_edge_index[1] = l2a_edge_index[1] + x_lane.shape[0]

        # conditioning vector for DiT block
        c = t + y + n
        # necessary for A2A and L2A attention
        c_small = self.downsample_c(c)

        # apply dropout
        x_lane = self.emb_drop(x_lane)
        x_agent = self.emb_drop(x_agent)

        # text conditioning
        text_context = None
        if self.cfg_model.text_conditioning and text_embeds is not None:
            text_context_lanes = self.text_embedding_lanes(text_embeds)   # [num_scenes, D_lane]
            text_context_agents = self.text_embedding_agents(text_embeds)  # [num_scenes, D_agent]
            if drop_mask.any():
                text_context_lanes[drop_mask] = 0
                text_context_agents[drop_mask] = 0
            text_context = (text_context_lanes, text_context_agents)

        # image conditioning
        img_tokens_lane = None
        img_tokens_agent = None
        if self.cfg_model.img_conditioning:
            depth_map = data['depth_map']
            dino_feats = data['dino_feats']
            # align depth map size with dino feat map size
            h, w = dino_feats.shape[2] * self.cfg_model.dino_patch_size, dino_feats.shape[3] * self.cfg_model.dino_patch_size
            depth_map = F.interpolate(depth_map, size=(h, w), mode='bilinear', align_corners=False)
            img_tokens_lane, img_tokens_agent = self._prepare_image_tokens(dino_feats, depth_map)
            if drop_mask.any():
                mask = drop_mask.view(-1, 1, 1)   # [B, 1, 1]
                img_tokens_lane = img_tokens_lane * (~mask)
                img_tokens_agent = img_tokens_agent * (~mask)

        # factorized dit block processing
        for block in self.blocks:
            x_lane, x_agent = block(
                x_lane,
                x_agent,
                c,
                c_small,
                l2l_edge_index,
                a2a_edge_index,
                l2a_edge_index,
                text_context=text_context if self.cfg_model.text_conditioning else None,
                img_lane_context=img_tokens_lane if self.cfg_model.img_conditioning else None,
                img_agent_context=img_tokens_agent if self.cfg_model.img_conditioning else None,
                lane_batch=data['lane'].batch,
                agent_batch=data['agent'].batch,
                )

        # decode the noise as in the original DiT paper
        c_lane = c[:x_lane.shape[0]]
        c_agent = c_small[x_lane.shape[0]:]
        x_lane = self.pred_lane_noise(x_lane, c_lane).unsqueeze(1)
        x_agent = self.pred_agent_noise(x_agent, c_agent).unsqueeze(1)

        return x_agent, x_lane
