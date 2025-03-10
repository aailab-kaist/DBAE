from functools import partial
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def uni_masks(k: int, latent_dim: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """setup masks for baseline model

    Args:
        k (int): # of segments
        latent_dim (int): dimension of latent code `z`
        device (str): in the form of f'cuda:{id}' or 'cpu'

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            The first tensor is `masks` used for sampling at correct timestep
            The second tensor is `k_masks` used for interpolating at specific segment
    """
    # setup masks
    masks = torch.zeros(k, latent_dim, device=device)
    k_masks = torch.zeros(k, latent_dim, device=device)
    for i in range(k):
        current_dim = int(latent_dim / k * (i + 1))
        prev_dim = int(latent_dim / k * i)
        k_masks[i, prev_dim:current_dim] = 1.0
        masks[i, 0:current_dim] = 1.0
    return (masks, k_masks)

def ueq_masks(k: int, latent_dim: int, device: str, ssl_config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """setup masks for uequal-dim model

    Args:
        k (int): # of segments
        latent_dim (int): dimension of latent code `z`
        device (str): in the form of f'cuda:{id}' or 'cpu'

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            The first tensor is `masks` used for sampling at correct timestep
            The second tensor is `k_masks` used for interpolating at specific segment
    """
    masks = torch.zeros(k, latent_dim, device=device)
    k_masks = torch.zeros(k, latent_dim, device=device)
    receding_masks = torch.zeros(k, latent_dim, device=device)

    # t_to_idx
    n_timesteps = 1000
    if 'stages' in ssl_config:
        stages = ssl_config["stages"].split(',')
        stages = [int(stage) for stage in stages]
    else:
        stages = None
    t_to_idx = torch.zeros(n_timesteps, device=device).long()
    if 'k_per_stage' in ssl_config:
        assert 'stages' in ssl_config
        k_per_stage = ssl_config["k_per_stage"].split(',')
        k_per_stage = [int(k) for k in k_per_stage]
        current_stage = 0
        sum_indices = 0
        for t in range(n_timesteps):
            if t == stages[current_stage]:
                sum_indices += k_per_stage[current_stage]
                current_stage += 1
            current_steps = float(stages[current_stage])
            current_k = float(k_per_stage[current_stage])
            t_to_idx[t] = int(float(t) / current_steps * current_k + sum_indices)
    else:
        for t in range(n_timesteps):
            t_to_idx[t] = int(float(t) / (float(n_timesteps) / k))
        k_per_stage = None

    # dims_per_stage
    if 'dims_per_stage' in ssl_config:
        dims_per_stage = ssl_config["dims_per_stage"].split(',')
        dims_per_stage = [int(k) for k in dims_per_stage]

    for i in range(k):
        #current_dim = int(float(latent_dim) / self.k * (i + 1))
        #prev_dim = int(float(latent_dim) / self.k * i)
        current_dim = get_mask_end_dim(i, latent_dim, t_to_idx, dims_per_stage, stages, k=k)
        prev_dim = get_mask_end_dim(i-1, latent_dim, t_to_idx, dims_per_stage, stages, k=k)
        masks[i, 0:current_dim] = 1.0
        k_masks[i, prev_dim:current_dim] = 1.0
        # receding_masks[i, 0:prev_dim] = 1.0
    return (masks, k_masks)

def get_mask_end_dim(idx, latent_dim, t_to_idx, dims_per_stage, stages, k=64):
    if idx >= k:
        assert False, 'max idx value is k-1!'
    if idx < 0:
        return 0
    if dims_per_stage is None:
        return int(float(latent_dim) / k * (idx+1))
    else:
        assert stages is not None, 'Config file error. No stages found under ssl_config!'
        # calculate how many blocks in total after each stage
        accum_num_blocks = np.zeros(len(stages))
        for i in range(k):
            start_t = torch.nonzero(t_to_idx==i, as_tuple=True)[0][0].item()
            stage = np.argmax(np.array(stages) > start_t)
            accum_num_blocks[stage:] += 1
        start_t = torch.nonzero(t_to_idx==idx, as_tuple=True)[0][0].item()
        stage = np.argmax(np.array(stages) > start_t)
        stage_total_dim = dims_per_stage[stage]
        stage_num_blocks = accum_num_blocks[stage] - accum_num_blocks[stage-1] if stage>0 else accum_num_blocks[stage]
        if int(accum_num_blocks[stage]) == idx + 1:
            return sum(dims_per_stage[0:stage+1])
        else:
            stage_prev_blocks = idx - int(accum_num_blocks[stage-1]) if stage>0 else idx
            return sum(dims_per_stage[0:stage]) + int(float(stage_total_dim) / float(stage_num_blocks)) * (stage_prev_blocks+1)


class HookHandler:
    def __init__(self):
        self.callback_fn = None

    def __call__(self, *args, **kwargs):
        return self.callback_fn(*args, **kwargs)

    def register(self, callback_fn):
        assert callable(callback_fn), "registered hook is not a function."
        self.callback_fn = callback_fn

    def remove(self):
        self.callback_fn = None
    
    def __bool__(self):
        return self.callback_fn is not None


class MaskedDiffusion:
    def __init__(self, config, device):
        super().__init__()
        self.device=device
        self.to_float = {"dtype": torch.float32, "device": device}  # common dtypes/devices for pre-computed constants
        self.timesteps = int(config["timesteps"])
        betas_type = config["betas_type"]
        if betas_type == "linear":
            self.betas = torch.linspace(config["linear_beta_start"], config["linear_beta_end"], self.timesteps, **self.to_float)
        else:
            raise NotImplementedError(f"only support linear beta scheduler, but got {betas_type}")

        # the following pre-computed constant tensors are all one-dimension
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        # define alphas_cumprod_0=1, complying with DDIM paper
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, **self.to_float), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        #######################################
        # below constants are not needed in DDIM. consider removing them in the future
        #######################################
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_recip_alphas_cumprod = (1 / self.alphas_cumprod).sqrt()
        self.sqrt_recip_alphas_cumprod_m1 = (1 / self.alphas_cumprod - 1).sqrt()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        # clip the log because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.cat([self.posterior_variance[1].view(1), self.posterior_variance[1:]]).log()
        self.x_0_posterior_mean_x_0_coef = self.betas * self.alphas_cumprod_prev.sqrt() / (1 - self.alphas_cumprod)
        self.x_0_posterior_mean_x_t_coef = self.alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.noise_posterior_mean_x_t_coef = (1 / self.alphas).sqrt()
        self.noise_posterior_mean_noise_coef = self.betas / (self.alphas.sqrt() * (1-self.alphas_cumprod).sqrt())
        self.shift_coef = - self.alphas.sqrt() * (1-self.alphas_cumprod_prev) / (1-self.alphas_cumprod).sqrt()

        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        gamma = 0.1
        self.weight = snr ** gamma / (1 + snr)
        #######################################

        self.hook = HookHandler()

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_seq(original_steps: int, new_steps: int)-> range:
        skip = original_steps // new_steps
        seq = range(0, original_steps, skip)
        return seq

    @torch.inference_mode()
    def guided_ddim_step(
            self, 
            decoder: nn.Module, 
            z: Optional[torch.Tensor], 
            x_t: torch.Tensor, 
            t: torch.Tensor, 
            next_t: torch.Tensor,
            use_guide: bool = True,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """single step ddim samping with supports:
            - masked `z` according to which diffusion step `z` is conditioning on.
            - guidance
            - forward ddim when `reverse_process=False`

        Args:
            decoder (nn.Module): mapping `x_t` into predicted noise with the guidance of `z`
            z (Optional[torch.Tensor]): shape (B, latent_dim).
                representations of x_0 for learning the guidance.
            x_t (torch.Tensor): shape (B,3,H,W)
            t (torch.Tensor): shape (B,)
            next_t (torch.Tensor): shape (B,)
            use_shift (bool): whether add guidance term to predicted noise. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            0) x_next of shape (B,3,H,W)
                when reverse_process=True, return `x_{t-1}`,
                else return `x_{t+1}`.
            1) predicted x0 of shape (B,3,H,W)
        """
        x_shape = x_t.shape
        assert t.shape == (x_shape[0],)
        assert next_t.shape == (x_shape[0],)
        if z is None:
            # guidance is disabled
            # pass dummy value to decoder to avoid modifying it
            z = torch.zeros((x_shape[0], decoder.latent_dim), **self.to_float)
        decoder.eval()
        predicted_noise, gradient = decoder(x_t, t, z)
        decoder.train()
        at = self.extract_coef_at_t(self.alphas_cumprod, t, x_shape)
        if next_t[0] == -1:
            at_next = torch.ones([x_shape[0]]+[1]*(len(x_shape)-1), **self.to_float)
        elif next_t[0] == -2:
            at_next = torch.zeros([x_shape[0]]+[1]*(len(x_shape)-1), **self.to_float)
        else:
            at_next = self.extract_coef_at_t(self.alphas_cumprod, next_t, x_shape)

        if use_guide:
            predicted_noise = predicted_noise - (1. - at).sqrt() * gradient

        predicted_x_0 = (1 / at).sqrt() * x_t - (1 / at - 1).sqrt() * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = ((1 / at).sqrt() * x_t - predicted_x_0) / (1 / at - 1).sqrt()

        x_next = at_next.sqrt() * predicted_x_0 + (1. - at_next).sqrt() * new_predicted_noise
        return (x_next, predicted_x_0)

    @torch.inference_mode()
    def masked_guided_ddim(
            self, 
            ddim_style: str, 
            encoder: Optional[nn.Module], 
            decoder: nn.Module, 
            x_0: Optional[torch.Tensor], 
            x_T: Optional[torch.Tensor], 
            z: Optional[torch.Tensor]   = None, 
            stop_percent: float         = 0.0, 
            reverse_process: bool       = True,
            masks: torch.Tensor         = None,
            verbose: bool               = True,
        ) -> torch.Tensor:
        """ddim sampling process with masks. 

        Args:
            ddim_style (str): str in the form of f"ddim{ddim_steps}"
            encoder (Optional[nn.Module]): mapping `x` into `z`.
                when `z` is not None, encoder is useless so can be set None safely.
            decoder (nn.Module): mapping `x_t` into predicted noise with the guidance of `z`
            x_0 (Optional[torch.Tensor]): shape (B, 3, H, W)
            x_T (Optional[torch.Tensor]): shape (B, 3, H, W)
            z (Optional[torch.Tensor], optional): shape (B, latent_dim).
                representations of x_0 for learning the guidance. Defaults to None.
                if z is None and encoder is None, disable guidance.
            stop_percent (float, optional): during sampling, stop using the guidance **after** timestep `stop_step`.
                stop_step = int(stop_percent * self.timesteps). Defaults to 0.0, using guided all the time.
                when stop_percent=1.0, disable guidance.
            reverse_process (bool, optional): whether reverse ddim (True) or forward ddim (False). Defaults to True.
            masks (torch.Tensor, optional): shape (k, latent_dim).
                Masks are performed on `z` according to which diffusion step `z` is conditioning on. 
                Defaults to None, which means no masks.

        Returns:
            torch.Tensor: x_next of shape (B,3,H,W)
        when reverse_process=True, return x_{t-1}, else return x_{t+1}.
        """
        if reverse_process:
            assert x_T is not None, "reverse ddim requires x_T."
            bs = x_T.shape[0]
        else:
            assert x_0 is not None, "forward ddim requires x_0."
            bs = x_0.shape[0]

        if z is None and encoder is None:
            if verbose:
                print("Disable z guidance...")
            use_guide = False
        else:
            if verbose:
                print("Enable z guidance...")
            use_guide = True
            if z is None:
                encoder.eval()
                z = encoder(x_0)
                encoder.train()
            assert z.shape[0] == bs

            if masks is not None:
                assert masks.shape[1]==z.shape[1], \
                    f"dims of masks ({masks.shape[1]}) do not match with z ({z.shape[1]})"

        num_ddim_steps = int(ddim_style[len("ddim"):])
        seq = MaskedDiffusion.get_seq(self.timesteps, num_ddim_steps)
        # minus numbers -1, -2 tag edge cases x_0 and x_T+1. see self.guided_ddim_step().
        if reverse_process:
            seq_next = [-1] + list(seq[:-1])
            img = x_T
        else:
            seq_next = list(seq[1:]) + [-2]
            img = x_0
        stop_step = int(stop_percent * self.timesteps)

        for i, j in tqdm(
            zip(reversed(seq), reversed(seq_next)) if reverse_process else zip(seq, seq_next), 
            desc="reverse ddim step" if reverse_process else "forward ddim step", 
            total=num_ddim_steps,
            # disable=None
        ):
            t = torch.full((bs,), i, device=self.device, dtype=torch.long)
            next_t = torch.full((bs,), j, device=self.device, dtype=torch.long)
            if z is not None:  # use guidance
                if i < stop_step:
                    z_t = None  # after stop step, disable guidance
                elif masks is not None:
                    # add masks to z
                    k = int(masks.shape[0])
                    idx = (t / (self.timesteps / k)).long()  # 0 ~ k-1
                    z_t = masks[idx] * z
                else:
                    z_t = z
            else:
                # disable guidance
                z_t = None

            img, pred_x0 = self.guided_ddim_step(
                decoder         = decoder, 
                z               = z_t, 
                x_t             = img, 
                t               = t,
                next_t          = next_t,
                use_guide       = use_guide,
            )

            self.call_hook(i, j, img, pred_x0)

        return img

    def call_hook(self, *args, **kargs):
        if self.hook:
            self.hook(*args, **kargs)

    def register_hook(self, hook: Callable[[int, int, torch.Tensor], None]) -> HookHandler:
        """
        register a hook after `guided_ddim_step` during ddim forward/reverse process.
        The hook should have the following signature:
            hook(t: int, t_next: int, x_next: torch.Tensor) -> Tensor or None
        The hook should not modify its argument.
        """
        assert callable(hook), "Error: registered hook is not a function."
        self.hook.register(hook)
        return self.hook

    """
        manipulation
    """
    def normalize(self, z, mean, std):
        return (z - mean) / std


    def denormalize(self, z, mean, std):
        return z * std + mean

    def manipulation_sample(self, ddim_style, classifier_weight, encoder, decoder, x_0, inferred_x_T, latents_mean, latents_std, class_id, scale, sample_mask=None):
        z = encoder(x_0)
        z_norm = self.normalize(z, latents_mean, latents_std)

        import math
        z_norm_manipulated = z_norm + scale * math.sqrt(512) * F.normalize(classifier_weight[class_id][None,:], dim=1)
        z_manipulated = self.denormalize(z_norm_manipulated, latents_mean, latents_std)

        # return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, inferred_x_T, z_manipulated, stop_percent=0.0)
        return self.masked_guided_ddim(ddim_style, None, decoder, None, inferred_x_T, z_manipulated, masks=sample_mask)
    
    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )
