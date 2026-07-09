import numpy as np
import torch
from torch import nn

from utils.diffusion_helpers import (
    cosine_beta_schedule,
    extract
)
from utils.losses import GeometricLosses
from nn_modules.dit import DiT
from nn_modules.text_encoders.umt5 import T5EncoderModel
from cfgs.config import BEFORE_PARTITION, UNIFIED_FORMAT_INDICES, NUPLAN_VEHICLE
from utils.torch_helpers import get_torch_dtype
from utils.data_helpers import diff_unnorm_scene_by_mode, unnormalize_latents
from models.scenario_control_autoencoder import ScenarioControlAutoEncoder

class LDM(nn.Module):
    def __init__(self, cfg, cfg_ae):
        super(LDM, self).__init__()

        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.model = DiT(cfg)

        if self.cfg_model.decode_in_training:
            self.autoencoder = ScenarioControlAutoEncoder.load_from_checkpoint(self.cfg_model.autoencoder_path, cfg=cfg_ae, map_location='cpu')
            # Freeze autoencoder weights
            for p in self.autoencoder.parameters():
                p.requires_grad = False

        self.dtype = get_torch_dtype(self.cfg.train.precision)
        n_timesteps = self.cfg_model.n_diffusion_timesteps
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.lane_sampling_temperature = self.cfg_model.lane_sampling_temperature

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        loss_type = self.cfg.train.loss_type
        self.lane_loss_fn = GeometricLosses[loss_type]((1,2))
        self.agent_loss_fn = GeometricLosses[loss_type]((1,2))

        # text conditioning
        self.text_conditioning = self.cfg_model.text_conditioning
        self.use_cached_text_embeds = self.cfg_dataset.get('use_cached_text_embeds', False)
        if self.text_conditioning and self.use_cached_text_embeds:
            print("Using cached text embeddings")

    def load_text_encoder(self):
        """Loads the T5 text encoder. Kept separate from __init__ since it's a large model
        that should only be materialized when text conditioning is enabled and cached
        embeddings aren't available."""
        if self.text_conditioning and not self.use_cached_text_embeds:
            text_encoder_cfg = self.cfg.text_encoder
            self.text_len = text_encoder_cfg.text_len
            self.text_encoder = T5EncoderModel(
                text_len=text_encoder_cfg.text_len,
                dtype=text_encoder_cfg.dtype,
                device=torch.device('cpu'),
                checkpoint_path=text_encoder_cfg.checkpoint_path,
                tokenizer_path=text_encoder_cfg.tokenizer,
            )
            print(f"Loaded text encoder from {text_encoder_cfg.checkpoint_path}")

    def predict_start_from_noise(self, x_t, t, noise):
        """ Predict the start of the diffusion chain from the noised sample x_t and noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def compute_vehicle_collision_penalty(
        self,
        agent_decoded,
        agent_types,
        agent_batch,
        agent_timesteps,
        tau=0.1,
        cap=1.0,
    ):
        """
        Collision penalty based on overlap area normalized by total surface area.
        Prevents large vehicles from receiving disproportionately large penalties.
        """
        device = agent_decoded.device
        total_penalty = torch.zeros(1, device=device)

        # Use or ignore filtering depending on config
        is_vehicle = (agent_types == NUPLAN_VEHICLE)
        filter_vehicle = self.cfg.train.get("collision_filter_for_vehicles", False)
        veh_states = agent_decoded[is_vehicle] if filter_vehicle else agent_decoded
        veh_batch = agent_batch[is_vehicle] if filter_vehicle else agent_batch
        veh_t = agent_timesteps[is_vehicle] if filter_vehicle else agent_timesteps

        if veh_states.numel() == 0:
            return total_penalty

        for b in veh_batch.unique():
            mask = veh_batch == b
            states = veh_states[mask]
            t_scene = veh_t[mask][0].float()  # scene timestep
            n = states.shape[0]
            if n < 2:
                continue

            timestep_weight = 1 - self.sqrt_one_minus_alphas_cumprod[t_scene.long()]

            centers = states[:, :2].clone()
            centers[0] = centers[0].detach()

            cos_h = states[:, UNIFIED_FORMAT_INDICES['cos_heading']].detach()
            sin_h = states[:, UNIFIED_FORMAT_INDICES['sin_heading']].detach()
            lengths = states[:, UNIFIED_FORMAT_INDICES['length']].detach()
            widths = states[:, UNIFIED_FORMAT_INDICES['width']].detach()

            # Rotation matrices
            R = torch.stack([
                torch.stack([cos_h, -sin_h], dim=-1),
                torch.stack([sin_h, cos_h], dim=-1)
            ], dim=-2)

            # Pairwise relative positions
            delta = centers[:, None, :] - centers[None, :, :]

            # Transform to local coordinate frame
            delta_local = torch.einsum("nij,nkj->nki", R, delta)
            dx_local = torch.abs(delta_local[..., 0])
            dy_local = torch.abs(delta_local[..., 1])

            # Overlapping region
            half_len = (lengths[:, None] + lengths[None, :]) / 2
            half_wid = (widths[:, None] + widths[None, :]) / 2

            overlap_x = torch.relu(half_len - dx_local)
            overlap_y = torch.relu(half_wid - dy_local)
            overlap_area = overlap_x * overlap_y

            # Remove self-pairs
            not_self = 1.0 - torch.eye(n, device=device)
            overlap_area = overlap_area * not_self

            # Normalization by surface area
            area = (lengths * widths)
            sum_area = area[:, None] + area[None, :]
            norm_overlap = overlap_area / (sum_area + 1e-6)

            # Smooth penalty
            penalty = torch.tanh(norm_overlap / tau)
            penalty = torch.clamp(penalty, max=cap)

            total_penalty += timestep_weight * penalty.sum() / (2.0 * n)

        return total_penalty

    def q_posterior(self, x_start, x_t, t):
        """ Compute the mean and log variance of the posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane, text_embeds=None, neg_text_embeds=None):
        """ Predict the mean and log variance of the posterior distribution p(x_{t-1} | x_t, x_0)."""
        # noise prediction
        conditional_epsilon_agent, conditional_epsilon_lane = self.model(x_agent, x_lane, data, t_agent, t_lane, text_embeds=text_embeds, unconditional=False, drop=False)
        if neg_text_embeds is None and text_embeds is not None:
            neg_text_embeds = torch.zeros_like(text_embeds)
        unconditional_epsilon_agent, unconditional_epsilon_lane = self.model(x_agent, x_lane, data, t_agent, t_lane, text_embeds=neg_text_embeds, unconditional=True, drop=False)
        # classifier-free guidance
        epsilon_agent = unconditional_epsilon_agent + self.cfg.train.guidance_scale * (conditional_epsilon_agent - unconditional_epsilon_agent)
        epsilon_lane = unconditional_epsilon_lane + self.cfg.train.guidance_scale * (conditional_epsilon_lane - unconditional_epsilon_lane)

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)

        # given the noise and timestep, predict the start of the diffusion chain
        x_agent_recon = self.predict_start_from_noise(x_agent, t=t_agent, noise=epsilon_agent)
        x_lane_recon = self.predict_start_from_noise(x_lane, t=t_lane, noise=epsilon_lane)

        # mean, log_var of the posterior distribution q(x_t-1 | x_t, x_0)
        model_mean_agent, posterior_log_variance_agent = self.q_posterior(x_start=x_agent_recon, x_t=x_agent, t=t_agent)
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(x_start=x_lane_recon, x_t=x_lane, t=t_lane)

        return model_mean_agent, posterior_log_variance_agent, model_mean_lane, posterior_log_variance_lane

    @torch.no_grad()
    def p_sample(self, x_agent, x_lane, data, t_agent, t_lane, text_embeds=None, neg_text_embeds=None, generator=None):
        """ Sample from the posterior distribution p(x_{t-1} | x_t, x_0)."""
        b_agent = t_agent.shape[0]
        b_lane = t_lane.shape[0]

        model_mean_agent, model_log_variance_agent, model_mean_lane, model_log_variance_lane = self.p_mean_variance(
            x_agent,
            x_lane,
            data,
            t_agent,
            t_lane,
            text_embeds,
            neg_text_embeds)

        noise_agent = torch.randn(
            x_agent.shape,
            device=x_agent.device,
            dtype=x_agent.dtype,
            generator=generator
        )
        noise_lane = torch.randn(
            x_lane.shape,
            device=x_lane.device,
            dtype=x_lane.dtype,
            generator=generator
        )

        # no noise when t == 0
        nonzero_mask_agent = (1 - (t_agent == 0).float()).reshape(b_agent, *((1,) * (len(x_agent.shape) - 1)))
        nonzero_mask_lane = (1 - (t_lane == 0).float()).reshape(b_lane, *((1,) * (len(x_lane.shape) - 1)))

        # sample from the posterior distribution using reparametrization trick
        next_x_agent = model_mean_agent + nonzero_mask_agent * (model_log_variance_agent).exp().sqrt() * noise_agent
        next_x_lane = model_mean_lane + nonzero_mask_lane * (model_log_variance_lane).exp().sqrt() * noise_lane * self.lane_sampling_temperature

        return next_x_agent, next_x_lane

    @torch.no_grad()
    def p_sample_loop(
        self,
        agent_shape,
        lane_shape,
        data,
        device='cuda',
        mode='initial_scene',
        return_diffusion_chain=False):
        """ Generate a batch of samples from the diffusion model."""

        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch
        batch_size = data.batch_size

        g = torch.Generator(device=device)
        g.manual_seed(self.cfg.eval.seed)

        x_agent = torch.randn(agent_shape, device=device, generator=g)
        # conditional generation on existing lane latents
        if mode == 'lane_conditioned':
            x_lane = data['lane'].latents[:, np.newaxis, :].to(device)
        # jointly generate lane and agent latents
        else:
            x_lane = torch.randn(lane_shape, device=device, generator=g) * self.lane_sampling_temperature

        # text conditioning
        text_embeds = None
        neg_text_embeds = None
        if self.text_conditioning and mode in ("initial_scene", "train"):
            if self.use_cached_text_embeds and data['text_embeds'] is not None:
                text_embeds = data['text_embeds'].to(self.dtype)
                batch_size = len(data['text'])
                text_embeds = text_embeds.view(batch_size, -1, text_embeds.shape[-1])
                neg_text_embeds = data['neg_text_embeds'].to(self.dtype) if 'neg_text_embeds' in data else None
                neg_text_embeds = neg_text_embeds.view(batch_size, -1, neg_text_embeds.shape[-1]) if neg_text_embeds is not None else None

            elif data['text'] is not None:
                with torch.no_grad():
                    text_embeds = self.text_encoder(data['text'], device=x_agent.device)
                    neg_text_embeds = self.text_encoder(data['neg_text'], device=x_agent.device) if 'neg_text' in data else None
                # pad to self.text_len
                text_embeds = torch.stack([
                    torch.cat(
                        [e, e.new_zeros(self.text_len - e.shape[0], e.shape[1])])
                    for e in text_embeds
                ])
                text_embeds = text_embeds.to(self.dtype)
                if neg_text_embeds is not None:
                    neg_text_embeds = torch.stack([
                        torch.cat(
                            [e, e.new_zeros(self.text_len - e.shape[0], e.shape[1])])
                        for e in neg_text_embeds
                    ])
                    neg_text_embeds = neg_text_embeds.to(self.dtype)

        # for sample visualizations during training, we can condition on the noiseless latents
        # before the partition to visualize inpainting performance.
        if mode == 'train':
            agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
            x_agent[agent_mask] = data['agent'].latents[agent_mask].unsqueeze(1)
            lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
            x_lane[lane_mask] = data['lane'].latents[lane_mask].unsqueeze(1)

        if mode == 'inpainting':
            cond_lane_mask = data['lane'].mask
            x_lane[cond_lane_mask] = data['lane'].latents[cond_lane_mask].unsqueeze(1)
            cond_agent_mask = data['agent'].mask
            x_agent[cond_agent_mask] = data['agent'].latents[cond_agent_mask].unsqueeze(1)

            # text conditioning is not yet supported for inpainting, so drop it
            if self.text_conditioning and not self.use_cached_text_embeds:
                text_embeds = torch.zeros(batch_size, self.text_len, self.cfg_model.text_feat_dim, device=x_lane.device)
                neg_text_embeds = torch.zeros(batch_size, self.text_len, self.cfg_model.text_feat_dim, device=x_lane.device)
            else:
                text_embeds = None
                neg_text_embeds = None

        # useful for cool visuals :)
        if return_diffusion_chain: diffusion_chain = [(x_agent, x_lane)]

        # simulate reverse diffusion chain
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = timesteps[agent_batch]
            t_lane = timesteps[lane_batch]

            x_agent, x_lane = self.p_sample(x_agent, x_lane, data, t_agent, t_lane, text_embeds, neg_text_embeds, generator=g)

            x_agent = torch.clip(x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)
            if mode == 'lane_conditioned':
                x_lane = data['lane'].latents[:, np.newaxis, :].to(device)
            else:
                # clip outputs to avoid degenerate samples
                x_lane = torch.clip(x_lane, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip)

            if mode == 'inpainting':
                cond_lane_mask = data['lane'].mask
                x_lane[cond_lane_mask] = data['lane'].latents[cond_lane_mask].unsqueeze(1)
                cond_agent_mask = data['agent'].mask
                x_agent[cond_agent_mask] = data['agent'].latents[cond_agent_mask].unsqueeze(1)

            if mode == 'train':
                agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
                x_agent[agent_mask] = data['agent'].latents[agent_mask].unsqueeze(1)
                lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
                x_lane[lane_mask] = data['lane'].latents[lane_mask].unsqueeze(1)

            if return_diffusion_chain: diffusion_chain.append((x_agent, x_lane))

        if return_diffusion_chain:
            return x_agent[:, 0], x_lane[:, 0], diffusion_chain
        else:
            return x_agent[:, 0], x_lane[:, 0]

    @torch.no_grad()
    def forward(self, data, mode='initial_scene'):
        """generate samples from the diffusion model"""

        agent_shape = data['agent'].x[:, np.newaxis, :].shape
        lane_shape = data['lane'].x[:, np.newaxis, :].shape

        return self.p_sample_loop(
            agent_shape,
            lane_shape,
            data,
            device=data['agent'].x.device,
            mode=mode,
            return_diffusion_chain=self.cfg.eval.return_diffusion_chain)


    def q_sample(self, x_start, t, noise=None):
        """generate noised sample for training"""
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(
            self,
            x_agent,
            x_lane,
            data,
            t_agent,
            t_lane):
        """ Compute the loss for the diffusion model."""

        # generate noised latents for training
        agent_noise = torch.randn_like(x_agent)
        x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        lane_noise = torch.randn_like(x_lane)
        x_lane_noisy = self.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)

        # for the partitioned scenes, condition on noiseless latents before partition
        agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
        x_agent_noisy[agent_mask] = x_agent[agent_mask]
        lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
        x_lane_noisy[lane_mask] = x_lane[lane_mask]

        # text conditioning
        text_embeds = None
        if self.text_conditioning:
            if self.use_cached_text_embeds and data['text_embeds'] is not None:
                text_embeds = data['text_embeds'].to(self.dtype)
                batch_size = len(data['text'])
                text_embeds = text_embeds.contiguous().view(batch_size, -1, text_embeds.shape[-1])

            elif data['text'] is not None:
                with torch.no_grad():
                    text_embeds = self.text_encoder(data['text'], device=x_agent.device)
                    # pad to self.text_len
                    text_embeds = torch.stack([
                        torch.cat(
                            [e, e.new_zeros(self.text_len - e.shape[0], e.shape[1])])
                        for e in text_embeds
                    ])
                    text_embeds = text_embeds.to(self.dtype)

        # dropout for classifier-free guidance of text conditioning
        if self.text_conditioning:
            drop = True
            drop_prob_text = self.cfg_model.get('drop_prob_text', 0.0)
        else:
            drop = False
            drop_prob_text = 0.0

        agent_noise_pred, lane_noise_pred = self.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane,
                                                       text_embeds=text_embeds, unconditional=False, drop=drop, drop_prob_text=drop_prob_text)

        assert agent_noise.shape == agent_noise_pred.shape
        assert lane_noise.shape == lane_noise_pred.shape

        # if lg_type == PARTITIONED and latent correspond to element BEFORE_PARTITION, no noise is added
        agent_mask = data['agent'].partition_mask == BEFORE_PARTITION
        agent_noise[agent_mask] = 0.

        agent_loss = self.agent_loss_fn(agent_noise_pred, agent_noise, data['agent'].batch)

        lane_mask = data['lane'].partition_mask == BEFORE_PARTITION
        lane_noise[lane_mask] = 0.
        lane_batch = data['lane'].batch
        lane_loss = self.lane_loss_fn(lane_noise_pred, lane_noise, lane_batch)

        loss = agent_loss + self.cfg.train.lane_weight * lane_loss

        lambda_collision = self.cfg.train.get('collision_weight', 0.0)
        if lambda_collision > 0.0 and self.cfg_model.decode_in_training:
            x_agent_recon = self.predict_start_from_noise(x_agent_noisy, t_agent, agent_noise_pred)
            x_lane_recon = self.predict_start_from_noise(x_lane_noisy, t_lane, lane_noise_pred)

            x_agent_recon = x_agent_recon.clone()
            x_agent_recon[agent_mask] = data['agent'].latents[agent_mask].unsqueeze(1)

            x_lane_recon = x_lane_recon.clone()
            x_lane_recon[lane_mask] = data['lane'].latents[lane_mask].unsqueeze(1)

            x_agent_recon_latents, x_lane_recon_latents = unnormalize_latents(
                x_agent_recon,
                x_lane_recon,
                self.cfg_dataset.agent_latents_mean,
                self.cfg_dataset.agent_latents_std,
                self.cfg_dataset.lane_latents_mean,
                self.cfg_dataset.lane_latents_std
                )

            agent_decoded, lane_decoded, agent_types, lane_types, lane_conn = \
                self.autoencoder.model.forward_decoder(
                    x_agent_recon_latents[:, 0],  # remove latent dimension
                    x_lane_recon_latents[:, 0],
                    data
                )

            # unnormalize the samples
            agent_decoded_unnormalized, lane_decoded_unnormalized = diff_unnorm_scene_by_mode(
                data, agent_decoded, lane_decoded, self.cfg_dataset
            )

            # compute collision regularization
            collision_penalty = self.compute_vehicle_collision_penalty(
                                        agent_decoded=agent_decoded_unnormalized,
                                        agent_types=agent_types,
                                        agent_batch=data['agent'].batch,
                                        agent_timesteps=t_agent,
                                        tau=0.1,
                                        cap=1.0
                                    )

            loss = loss + lambda_collision * collision_penalty

            return loss, agent_loss, lane_loss, collision_penalty
        else:
            return loss, agent_loss, lane_loss


    def loss(self, data):
        """ Sample diffusion timesteps for training and compute the loss for the diffusion model."""
        # batch of agent and lane latents
        x_agent = data['agent'].latents.unsqueeze(1)
        x_lane = data['lane'].latents.unsqueeze(1)

        agent_batch = data['agent'].batch
        lane_batch = data['lane'].batch
        batch_size = data.batch_size

        # batch of random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_agent.device).long()
        t_agent = t[agent_batch]
        t_lane = t[lane_batch]

        if self.cfg.train.get('collision_weight', 0.0) > 0.0:
            loss, agent_loss, lane_loss, collision_penalty = self.p_losses(x_agent, x_lane, data, t_agent, t_lane)

            loss_dict = {
                'loss': loss.mean(),
                'agent_loss': agent_loss.mean().detach(),
                'lane_loss': lane_loss.mean().detach(),
                'collision_penalty': collision_penalty.mean().detach()
            }
        else:
            loss, agent_loss, lane_loss = self.p_losses(x_agent, x_lane, data, t_agent, t_lane)
            loss_dict = {
                'loss': loss.mean(),
                'agent_loss': agent_loss.mean().detach(),
                'lane_loss': lane_loss.mean().detach(),
            }

        return loss_dict


    @torch.no_grad()
    def compute_collision_over_timesteps(self, data, autoencoder=None):
        """
        Computes collision penalty for every timestep t in [0, T-1].
        Used for visualization during evaluation.
        """
        device = data['agent'].latents.device
        batch_size = data.batch_size

        # original latents
        x_agent = data['agent'].latents.unsqueeze(1)
        x_lane = data['lane'].latents.unsqueeze(1)

        collision_curve = []

        for t_scalar in range(self.n_timesteps):
            # same timestep for entire batch
            t = torch.full((batch_size,), t_scalar, device=device, dtype=torch.long)
            t_agent = t[data['agent'].batch]
            t_lane = t[data['lane'].batch]

            # rebuild x_t = sqrt(a)*x0 + sqrt(1-a)*noise
            agent_noise = torch.randn_like(x_agent)
            x_agent_noisy = self.q_sample(x_agent, t_agent, agent_noise)

            lane_noise = torch.randn_like(x_lane)
            x_lane_noisy = self.q_sample(x_lane, t_lane, lane_noise)

            # predict noise
            agent_noise_pred, lane_noise_pred = self.model(
                x_agent_noisy, x_lane_noisy, data,
                t_agent, t_lane,
                unconditional=False, drop=False
            )

            # reconstruct x0_hat
            x_agent_recon = self.predict_start_from_noise(x_agent_noisy, t_agent, agent_noise_pred)
            x_lane_recon = self.predict_start_from_noise(x_lane_noisy, t_lane, lane_noise_pred)

            # unnormalize latents
            x_agent_lat, x_lane_lat = unnormalize_latents(
                x_agent_recon,
                x_lane_recon,
                self.cfg_dataset.agent_latents_mean,
                self.cfg_dataset.agent_latents_std,
                self.cfg_dataset.lane_latents_mean,
                self.cfg_dataset.lane_latents_std
            )

            # decode
            decoder = autoencoder if autoencoder is not None else self.autoencoder
            agent_dec, lane_dec, agent_types, lane_types, lane_conn = \
                decoder.model.forward_decoder(
                    x_agent_lat[:, 0], x_lane_lat[:, 0], data
                )

            # unnormalize states
            agent_dec_unnorm, lane_dec_unnorm = diff_unnorm_scene_by_mode(
                data, agent_dec, lane_dec, self.cfg_dataset
            )

            # compute collision penalty
            col = self.compute_vehicle_collision_penalty(
                agent_decoded=agent_dec_unnorm,
                agent_types=agent_types,
                agent_batch=data['agent'].batch,
                agent_timesteps=t_agent,
                tau=0.1,
                cap=1.0,
            )

            collision_curve.append(col.item())

        return collision_curve
