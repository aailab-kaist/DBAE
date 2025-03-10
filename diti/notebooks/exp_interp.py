from PIL import Image
import torch
from typing import Optional, List
from enum import Enum

import dataset as dataset_module
from model.diffusion import GaussianDiffusion
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from utils import load_yaml, move_to_cuda
from eval.src.eval_utils import ImageCanvas
from eval.src.masked_diffusion import uni_masks


def calculate_theta(a,b):
    return torch.arccos(torch.dot(a.view(-1),b.view(-1))/(torch.norm(a)*torch.norm(b)))

def slerp(a,b,alpha):
    theta = calculate_theta(a,b)
    sin_theta = torch.sin(theta)
    return a * torch.sin((1.0 - alpha)*theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta

def lerp(a,b,alpha):
    return (1.0 - alpha) * a + alpha * b

class Interp:
    def __init__(self, config, device, mask_tuple, f=1000, r=100):
        self.device = device
        torch.cuda.set_device(device)
        self.f, self.r = f, r

        config_path = config["config_path"]
        checkpoint_path = config["checkpoint_path"]
        model_config = load_yaml(config_path)

        self.k = 64
        self.latent_dim = model_config["encoder_config"]["latent_dim"]
        # build models
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=device)
        self.encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
        try:
            trained_ddpm_config = load_yaml(model_config["trained_ddpm_config"])
        except FileNotFoundError:
            # delete '../'
            trained_ddpm_config = load_yaml(model_config["trained_ddpm_config"][3:])
        self.decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])

        # load state dict
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        self.encoder.load_state_dict(checkpoint['ema_encoder'])
        self.decoder.load_state_dict(checkpoint['ema_decoder'])

        self.encoder.cuda().eval()
        self.decoder.cuda().eval()

        # dataset
        dataset_name = config["dataset_name"]
        self.dataset = getattr(dataset_module, dataset_name, None)(config["dataset_kwargs"])
        image_size = config["dataset_kwargs"]["image_size"]
        self.image_size = image_size

        self.masks, self.k_masks = mask_tuple
        self.image_index_1 = config["image_index_1"]
        self.image_index_2 = config["image_index_2"]
        data_1 = self.dataset.__getitem__(self.image_index_1)
        x_0_1 = move_to_cuda(data_1["x_0"]).unsqueeze(0)
        self.gt_1 = data_1["gt"]

        data_2 = self.dataset.__getitem__(self.image_index_2)
        x_0_2 = move_to_cuda(data_2["x_0"]).unsqueeze(0)
        self.gt_2 = data_2["gt"]

        with torch.inference_mode():
            z = self.encoder(torch.cat([x_0_1, x_0_2], dim=0))
            x_T = self.gaussian_diffusion.masked_guided_ddim(
                f'ddim{self.f}', None, self.decoder, torch.cat([x_0_1, x_0_2]), None, z, reverse_process=False, masks=self.masks, verbose=False
            )

        self.x_T_1 = x_T[0:1]
        self.x_T_2 = x_T[1:2]
        self.z_1 = z[0:1]
        self.z_2 = z[1:2]

    @torch.inference_mode()
    def lerp_seg(self, i, j, alpha_range=[0, 1], opp_dir=True, lerpT=True):

        height = 2
        width = len(alpha_range)
        merge = Image.new('RGB', (width * self.image_size, height * self.image_size), color=(255, 255, 255))

        k_mask_cum = torch.zeros((self.latent_dim,), device=self.device)
        for m in range(max(i, 0), min(j, self.k)):
            k_mask_cum += self.k_masks[m]

        z1_slice = self.z_1 * k_mask_cum.unsqueeze(0)
        z2_slice = self.z_2 * k_mask_cum.unsqueeze(0)

        # lerp z
        for i, alpha in enumerate(alpha_range):
            if lerpT:
                x_T = slerp(self.x_T_1, self.x_T_2, alpha)
            else:
                x_T = self.x_T_1

            k_lerp = lerp(z1_slice, z2_slice, alpha)
            # lerp k_i, fix other channels in image 1
            z12 = self.z_1.clone()
            z12 = self.z_1 - z1_slice + k_lerp

            # image = gaussian_diffusion.masked_representation_learning_ddim_sample(None, f'ddim100', None, decoder, None, x_T, z12)
            image = self.gaussian_diffusion.masked_guided_ddim(
                f'ddim{self.r}', None, self.decoder, None, x_T, z12, masks=self.masks, verbose=False
            )

            image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            merge.paste(Image.fromarray(image[0]), (i * self.image_size, 0))

            # lerp k_i, fix other channels in image 2
            if not opp_dir:
                x_T = slerp(self.x_T_2, self.x_T_1, alpha)
                k_lerp = lerp(z2_slice, z1_slice, alpha)
            if not lerpT:
                x_T = self.x_T_2
            z21 = self.z_2.clone()
            z21 = self.z_2 - z2_slice + k_lerp

            # image = gaussian_diffusion.masked_representation_learning_ddim_sample(None, f'ddim100', None, decoder, None, x_T, z21)
            image = self.gaussian_diffusion.masked_guided_ddim(
                f'ddim{self.r}', None, self.decoder, None, x_T, z21, masks=self.masks, verbose=False
            )

            image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            merge.paste(Image.fromarray(image[0]), (i * self.image_size, self.image_size))
        return merge

    @torch.inference_mode()
    def lerp_full(self, alpha_range=[0, 1], opp_dir=True, lerpT=True):

        height = 1
        width = len(alpha_range)
        merge = Image.new('RGB', (width * self.image_size, height * self.image_size), color=(255, 255, 255))

        # lerp z
        for i, alpha in enumerate(alpha_range):
            if lerpT:
                x_T = slerp(self.x_T_1, self.x_T_2, alpha)
            else:
                x_T = self.x_T_1

            k_lerp = lerp(self.z_1, self.z_2, alpha)

            image = self.gaussian_diffusion.masked_guided_ddim(
                f'ddim{self.r}', None, self.decoder, None, x_T, k_lerp, masks=self.masks, verbose=False
            )

            image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            merge.paste(Image.fromarray(image[0]), (i * self.image_size, 0))

            # # lerp k_i, fix other channels in image 2
            # if not opp_dir:
            #     x_T = slerp(self.x_T_2, self.x_T_1, alpha)
            #     k_lerp = lerp(z2_slice, z1_slice, alpha)
            # if not lerpT:
            #     x_T = self.x_T_2
            # z21 = self.z_2.clone()
            # z21 = self.z_2 - z2_slice + k_lerp

            # # image = gaussian_diffusion.masked_representation_learning_ddim_sample(None, f'ddim100', None, decoder, None, x_T, z21)
            # image = self.gaussian_diffusion.masked_guided_ddim(
            #     f'ddim{self.r}', None, self.decoder, None, x_T, z21, masks=self.masks, verbose=False
            # )

            # image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            # image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            # merge.paste(Image.fromarray(image[0]), (i * self.image_size, self.image_size))
        return merge

    def exp_lerp_simple(self, loc = [0], wid = [64], alpha_range = [0, 1/3, 2/3, 1.]):
        img_txt_list = []
        assert len(loc) == len(wid), f"len(loc) ({len(loc)}) != len(wid) ({len(wid)})"
        
        if loc == [0] and wid == [64]:
            # full dim interpolation
            img_txt_list.append((self.lerp_full(alpha_range), f"full dim interpolation"))
        else:
            # segment interpolation
            for i, w in zip(loc, wid):
                img_txt_list.append((self.lerp_seg(i, i+w, alpha_range), f"{i}->{i+w} (w{w})"))

        # visualization
        merge = ImageCanvas.build_from_list(
            img_txt_list, 
            img_txt_list[0][0].size, 
            num_text_image_rows=len(img_txt_list),
            text_image_height_ratio=0.2,
            txt_shift=0.3)
        return merge


class Method(Enum):
    FFHQ_PDAE = "ffhq_pdae"
    FFHQ_DITI = "ffhq_diti"

def interpolate(
    gpu: int = 0,
    img1: int = None,
    img2: int = None,
    split: str = "train",
    f: int = 1000,
    r: int = 100,
    mask_z: bool = True,
    ckpt_root: str = 'path/to/ckpt_root_1',
    ckpt_name: str = 'latest',
    data_path: str = 'path/to/data_path',
    loc: List[int] = [0], 
    wid: List[int] = [64],
    alpha_range: List[float] = [0, 1/3, 2/3, 1.],
):
    if img1 is None or img2 is None:
        raise ValueError("Both img1 and img2 must be specified.")
    if not isinstance(img1, int) or not isinstance(img2, int):
        raise TypeError("img1 and img2 must be integers.")

    device = f"cuda:{gpu}"

    config = {
        "config_path": f"{ckpt_root}/config.yml",
        "checkpoint_path": f"{ckpt_root}/checkpoints/{ckpt_name}.pt",

        "dataset_name": "FFHQ",
        "dataset_kwargs": {
            "data_path": data_path,
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False, 
            "split": split,
        },
        "image_index_1": img1,
        "image_index_2": img2,
    }
    k = 64
    latend_dim = 512
    mask_tuple = uni_masks(k, latend_dim, device)
    if not mask_z: 
        mask_tuple = (None, mask_tuple[1])

    interp = Interp(config, device, mask_tuple, f=f, r=r)
    merge = interp.exp_lerp_simple(loc = loc, wid = wid, alpha_range = alpha_range)
    return merge
