from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .ddim import DDIM


class GaussianDiffusion:
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

        #######################################
        # below constants are not needed in DDIM. consider removing them in the future
        #######################################
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
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


    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        original_timesteps = original_alphas_cumprod.shape[0]
        dim_step = int(ddim_style[len("ddim"):])
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, dim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)

    # x_start: batch_size x channel x height x width
    # t: batch_size
    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def q_posterior_mean(self, x_0, x_t, t):
        shape = x_t.shape
        return self.extract_coef_at_t(self.x_0_posterior_mean_x_0_coef, t, shape) * x_0 \
               + self.extract_coef_at_t(self.x_0_posterior_mean_x_t_coef, t, shape) * x_t

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def noise_p_sample(self, x_t, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def x_0_clip_p_sample(self, x_t, t, predicted_noise, learned_range=None, clip_x_0=True):
        shape = x_t.shape

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
        if clip_x_0:
            predicted_x_0.clamp_(-1,1)
        predicted_mean = self.q_posterior_mean(predicted_x_0, x_t, t)
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def learned_range_to_log_variance(self, learned_range, t):
        shape = learned_range.shape
        min_log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        max_log_variance = self.extract_coef_at_t(torch.log(self.betas), t, shape)
        # The learned_range is [-1, 1] for [min_var, max_var].
        frac = (learned_range + 1) / 2
        return min_log_variance + frac * (max_log_variance - min_log_variance)

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
               - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def predicted_noise_to_predicted_mean(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
               self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise

    def p_loss(self, noise, predicted_noise, weight=None, loss_type="l2", mean=True):
        if loss_type == 'l1':
            loss = (noise - predicted_noise).abs()
        elif loss_type == 'l2':
            if weight is not None:
                loss = weight * (noise - predicted_noise) ** 2
            else:
                loss = (noise - predicted_noise) ** 2
        else:
            raise NotImplementedError()
        if mean:
            return loss.mean()
        else:
            return loss

    """
        test pretrained dpms
    """
    def test_pretrained_dpms(self, ddim_style, denoise_fn, x_T):
        return self.ddim_sample(ddim_style, denoise_fn, x_T)

    """
        ddim
    """
    def ddim_sample(self, ddim_style, denoise_fn, x_T):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(denoise_fn, x_T)

    def ddim_encode(self, ddim_style, denoise_fn, x_0):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_encode_loop(denoise_fn, x_0)

    """
        regular
    """
    def regular_train_one_batch(self, denoise_fn, x_0):
        shape = x_0.shape
        batch_size = shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_t, t)

        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def regular_ddim_sample(self, ddim_style, denoise_fn, x_T):
        return self.ddim_sample(ddim_style, denoise_fn, x_T)

    def regular_ddpm_sample(self, denoise_fn, x_T):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = denoise_fn(img, t, None)
            img = self.noise_p_sample(img, t, predicted_noise)
        return img

    """
        representation learning
    """
    def representation_learning_train_one_batch(self, encoder, decoder, x_0):
        shape = x_0.shape
        batch_size = shape[0]

        z = encoder(x_0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        predicted_noise, gradient = decoder(x_t, t, z)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)

        # weight = None
        weight = self.extract_coef_at_t(self.weight, t ,shape)

        prediction_loss = self.p_loss(noise, predicted_noise + shift_coef * gradient, weight=weight)

        return {
            'prediction_loss': prediction_loss
        }
    
    '''
        PDAE + timestep stratification
    '''
    def pdae_timestep_loss(self, encoder, decoder, x_0, t_to_idx, masks, k_masks, receding_masks, mode, pdae_z=None):
        shape = x_0.shape
        bs = shape[0]
        k = masks.shape[0]

        if pdae_z is None:
            z = encoder(x_0)
        else:
            z = pdae_z

        t = torch.randint(0, self.timesteps, (bs,), device=self.device, dtype=torch.long)
        if t_to_idx is None:
            idx = (t.float() / (float(self.timesteps) / k)).long()    # 0 ~ k-1
        else:
            idx = t_to_idx[t]
        

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        if mode in ['base', 'dual']:
            masks_batch = masks[idx]
            predicted_noise, gradient = decoder(x_t, t, z * masks_batch)
        if mode in ['stop_grad', 'dual']:
            k_masks_batch = k_masks[idx]
            receding_masks_batch = receding_masks[idx]
            cond_emb = z.detach() * receding_masks_batch + z * k_masks_batch
            predicted_nograd, gradient_nograd = decoder(x_t, t, cond_emb)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
        weight = self.extract_coef_at_t(self.weight, t ,shape)

        prediction_loss = None
        if mode in ['base', 'dual']:
            prediction_loss = self.p_loss(noise, predicted_noise + shift_coef * gradient, weight=weight, mean=False)
        if mode in ['stop_grad', 'dual']:
            if prediction_loss is not None:
                prediction_loss += self.p_loss(noise, predicted_nograd + shift_coef * gradient_nograd, weight=weight, mean=False)
            else:
                prediction_loss = self.p_loss(noise, predicted_nograd + shift_coef * gradient_nograd, weight=weight, mean=False)
        return prediction_loss.mean(), prediction_loss, t

    def representation_learning_ddpm_sample(self, encoder, decoder, x_0, x_T, z=None):
        shape = x_0.shape
        batch_size = shape[0]

        if z is None:
            z = encoder(x_0)
        img = x_T

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise, gradient = decoder(img, t, z)
            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            img = self.noise_p_sample(img, t, predicted_noise + shift_coef * gradient)
        return img


    def representation_learning_ddim_sample(self, ddim_style, encoder, decoder, x_0, x_T, z=None, stop_percent=0.0):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_sample_loop(decoder, z, x_T, stop_percent=stop_percent)

    @staticmethod
    def get_seq(original_steps: int, new_steps: int)-> range:
        skip = original_steps // new_steps
        seq = range(0, original_steps, skip)
        return seq

    @torch.inference_mode()
    def guided_ddim_step(
            self, 
            decoder: nn.Module, 
            z: torch.Tensor, 
            x_t: torch.Tensor, 
            t: torch.Tensor, 
            next_t: torch.Tensor,
        ) -> torch.Tensor:
        """single step ddim samping with supports:
            - masked `z` according to which diffusion step `z` is conditioning on.
            - guidance
            - forward ddim when `reverse_process=False`

        Args:
            decoder (nn.Module): mapping `x_t` into predicted noise with the guidance of `z`
            z (torch.Tensor): shape (B, latent_dim).
                representations of x_0 for learning the guidance.
            x_t (torch.Tensor): shape (B,3,H,W)
            t (torch.Tensor): shape (B,)
            next_t (torch.Tensor): shape (B,)
            reverse_process (bool, optional): whether reverse ddim (True) or forward ddim (False). Defaults to True.

        Returns:
            torch.Tensor: x_next of shape (B,3,H,W)
                when reverse_process=True, return `x_{t-1}`,
                else return `x_{t+1}`.
        """
        x_shape = x_t.shape
        assert t.shape == (x_shape[0],)
        assert next_t.shape == (x_shape[0],)
        if z is None:
            # guidance is disabled
            # send dummy value to decoder to avoid modifying it
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

        if z is not None:
            predicted_noise = predicted_noise - (1. - at).sqrt() * gradient

        predicted_x_0 = (1 / at).sqrt() * x_t - (1 / at - 1).sqrt() * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = ((1 / at).sqrt() * x_t - predicted_x_0) / (1 / at - 1).sqrt()

        return at_next.sqrt() * predicted_x_0 + (1. - at_next).sqrt() * new_predicted_noise

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
        else:
            if verbose:
                print("Enable z guidance...")
            if z is None:
                encoder.eval()
                z = encoder(x_0)
                encoder.train()
            assert z.shape[0] == bs

            if masks is not None:
                assert masks.shape[1]==z.shape[1], \
                    f"dims of masks ({masks.shape[1]}) do not match with z ({z.shape[1]})"

        num_ddim_steps = int(ddim_style[len("ddim"):])
        seq = GaussianDiffusion.get_seq(self.timesteps, num_ddim_steps)
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

            img = self.guided_ddim_step(
                decoder         = decoder, 
                z               = z_t, 
                x_t             = img, 
                t               = t,
                next_t          = next_t,
            )
        return img

    def representation_learning_ddim_encode(self, ddim_style, encoder, decoder, x_0, z=None):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_encode_loop(decoder, z, x_0)

    def representation_learning_autoencoding(self, encoder_ddim_style, decoder_ddim_style, encoder, decoder, x_0):
        z = encoder(x_0)
        inferred_x_T = self.representation_learning_ddim_encode(encoder_ddim_style, encoder, decoder, x_0, z)
        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, inferred_x_T, z)

    def representation_learning_gap_measure(self, encoder, decoder, x_0):
        shape = x_0.shape
        batch_size = shape[0]
        z = encoder(x_0)

        predicted_posterior_mean_gap = []
        autoencoder_posterior_mean_gap = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.q_sample(x_0, t, torch.rand_like(x_0))
            predicted_noise, gradient = decoder(x_t, t, z)

            predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
            predicted_posterior_mean = self.q_posterior_mean(predicted_x_0, x_t, t)

            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
            autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)
            autoencoder_predicted_posterior_mean = self.q_posterior_mean(autoencoder_predicted_x_0, x_t, t)

            true_posterior_mean = self.q_posterior_mean(x_0, x_t, t)

            predicted_posterior_mean_gap.append(torch.mean((true_posterior_mean - predicted_posterior_mean) ** 2, dim=[0, 1, 2, 3]).cpu().item())
            autoencoder_posterior_mean_gap.append(torch.mean((true_posterior_mean - autoencoder_predicted_posterior_mean) ** 2, dim=[0, 1, 2, 3]).cpu().item())

        return predicted_posterior_mean_gap, autoencoder_posterior_mean_gap

    def representation_learning_denoise_one_step(self, encoder, decoder, x_0, timestep_list):
        shape = x_0.shape

        t = torch.tensor(timestep_list, device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_0, t, noise=torch.randn_like(x_0))
        z = encoder(x_0)
        predicted_noise, gradient = decoder(x_t, t, z)

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
        autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
        autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)

        return predicted_x_0, autoencoder_predicted_x_0

    def representation_learning_ddim_trajectory_interpolation(self, ddim_style, decoder, z_1, z_2, x_T, alpha):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_trajectory_interpolation(decoder, z_1, z_2, x_T, alpha)

    """
        latent
    """
    @property
    def latent_diffusion_config(self):
        timesteps = 1000
        betas = np.array([0.008] * timesteps)
        # betas = np.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        loss_type = "l1"

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        return {
            "timesteps": timesteps,
            "betas": betas,
            "alphas_cumprod": to_torch(alphas_cumprod),
            "sqrt_alphas_cumprod": to_torch(sqrt_alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": to_torch(sqrt_one_minus_alphas_cumprod),
            "loss_type": loss_type,
        }

    def normalize(self, z, mean, std):
        return (z - mean) / std


    def denormalize(self, z, mean, std):
        return z * std + mean


    def latent_diffusion_train_one_batch(self, latent_denoise_fn, encoder, x_0, latents_mean, latents_std):
        timesteps = self.latent_diffusion_config["timesteps"]

        sqrt_alphas_cumprod = self.latent_diffusion_config["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod = self.latent_diffusion_config["sqrt_one_minus_alphas_cumprod"]

        z_0 = encoder(x_0)
        z_0 = z_0.detach()
        z_0 = self.normalize(z_0, latents_mean, latents_std)

        shape = z_0.shape
        batch_size = shape[0]

        t = torch.randint(0, timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_0)

        z_t = self.extract_coef_at_t(sqrt_alphas_cumprod, t, shape) * z_0 \
              + self.extract_coef_at_t(sqrt_one_minus_alphas_cumprod, t, shape) * noise

        predicted_noise = latent_denoise_fn(z_t, t)

        prediction_loss = self.p_loss(noise, predicted_noise, loss_type=self.latent_diffusion_config["loss_type"])

        return {
            'prediction_loss': prediction_loss,
        }

    def latent_diffusion_sample(self, latent_ddim_style, decoder_ddim_style, latent_denoise_fn, decoder, x_T, latents_mean, latents_std):
        alphas_cumprod = self.latent_diffusion_config["alphas_cumprod"]

        batch_size = x_T.shape[0]
        input_channel = latent_denoise_fn.module.input_channel
        z_T = torch.randn((batch_size, input_channel), device=self.device)

        z_T.clamp_(-1.0, 1.0) # may slightly improve sample quality

        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(latent_ddim_style, alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        z = ddim.latent_ddim_sample_loop(latent_denoise_fn, z_T)

        z = self.denormalize(z, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, x_T, z, stop_percent=0.3)


    """
        manipulation
    """

    def manipulation_sample(self, ddim_style, classifier_weight, encoder, decoder, x_0, inferred_x_T, latents_mean, latents_std, scale):
        z = encoder(x_0)
        z_norm = self.normalize(z, latents_mean, latents_std)

        import math
        z_norm_manipulated = z_norm + scale * math.sqrt(512) * F.normalize(classifier_weight[None,:], dim=1)
        z_manipulated = self.denormalize(z_norm_manipulated, latents_mean, latents_std)

        # return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, inferred_x_T, z_manipulated, stop_percent=0.0)
        return self.masked_guided_ddim(ddim_style, None, decoder, None, inferred_x_T, z_manipulated)
