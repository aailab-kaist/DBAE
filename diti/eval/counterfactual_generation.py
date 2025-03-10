""" sampling with interpolated z
discuss how many channels of z we interpolate could lead to significant change of the interpolated image
"""
import sys
sys.path.append("../")

from PIL import Image
import torch
from pathlib import Path
from typing import Optional
import contextlib, time
import fire
from enum import Enum

import dataset as dataset_module
from model.diffusion import GaussianDiffusion
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from utils import load_yaml, move_to_cuda
from eval.src.eval_utils import ImageCanvas
from eval.src.masked_diffusion import uni_masks, ueq_masks
import os

# utils
@contextlib.contextmanager
def context_timer():
    start_time = time.time()
    # print(message+"...")
    try:
        yield
    finally:
        duration = time.time() - start_time
        duration_msg = f"{int(duration//60):02d}m {duration%60:02.2f}s"
        print(duration_msg)

def calculate_theta(a,b):
    return torch.arccos(torch.dot(a.view(-1),b.view(-1))/(torch.norm(a)*torch.norm(b)))

def slerp(a,b,alpha):
    theta = calculate_theta(a,b)
    sin_theta = torch.sin(theta)
    return a * torch.sin((1.0 - alpha)*theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta

def lerp(a,b,alpha):
    return (1.0 - alpha) * a + alpha * b

class Interp:
    def __init__(self, config, alpha_range, device, mask_tuple, f=1000, r=100):
        self.alpha_range = alpha_range
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
        # gt_1 = data_1["gt"]

        data_2 = self.dataset.__getitem__(self.image_index_2)
        x_0_2 = move_to_cuda(data_2["x_0"]).unsqueeze(0)
        # gt_2 = data_2["gt"]

        with torch.inference_mode():
            z = self.encoder(torch.cat([x_0_1, x_0_2], dim=0))
            x_T = self.gaussian_diffusion.masked_guided_ddim(
                f'ddim{self.f}', None, self.decoder, torch.cat([x_0_1, x_0_2]), None, z, reverse_process=False, masks=self.masks
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
                f'ddim{self.r}', None, self.decoder, None, x_T, z12, masks=self.masks
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
                f'ddim{self.r}', None, self.decoder, None, x_T, z21, masks=self.masks
            )

            image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            merge.paste(Image.fromarray(image[0]), (i * self.image_size, self.image_size))
        return merge

    def exp_lerp_simple(self):
        loc = [2, 16, 24, 32]
        wid = [4, 8, 8, 32]
        img_txt_list = []
        assert len(loc) == len(wid), f"len(loc) ({len(loc)}) != len(wid) ({len(wid)})"
        alpha_range = [0, .25, .5, .75, 1.]
        for i, w in zip(loc, wid):
            img_txt_list.append((self.lerp_seg(i, i+w, alpha_range), f"{i}->{i+w} (w{w})"))

        # visualization
        merge = merge = ImageCanvas.build_from_list(
            img_txt_list, 
            img_txt_list[0][0].size, 
            num_text_image_rows=len(img_txt_list),
            text_image_height_ratio=0.2,
            txt_shift=0.3
        )
        return merge

    def exp_last_whentochange(self):
        k = 64
        loc = [32, 36, 40, 44, 48, 52, 56, 60]
        img_txt_list = []
        for i in loc:
            img_txt_list.append((self.lerp_seg(i, k), f"{i}->{k}"))

        # visualization
        merge = ImageCanvas.build_from_list(img_txt_list, img_txt_list[0][0].size, num_text_image_rows=len(img_txt_list))
        return merge


def main(
        gpu: int = 0,
        ckpt_root: str = "",
        ckpt_name: str = "latest",
        img1: Optional[int] = None,
        img2: Optional[int] = None,
        split: Optional[int] = None,
        f: int = 1000,
        r: int = 100,
        dataset: str = "celeba",
        output_dir: str = "../runs/cf_generations"
):

    k = 64
    latend_dim = 512
    # k_range = list(range(0,k-(w-1), 18))
    alpha_range = torch.linspace(0,1,2).tolist()
    device = f"cuda:{gpu}"
    
    method_name = os.path.basename(os.path.normpath(ckpt_root))
    print(f"Counterfactual generation between image {img1} and image {img2} in {split} split using method {method_name}...")

    config = {
        "config_path": os.path.join(ckpt_root, "config.yml"),
        "checkpoint_path": os.path.join(ckpt_root, "checkpoints", f"{ckpt_name}.pt")
    }
    if dataset == "celeba":
        config.update({
            "dataset_name": "CELEBA64",
            "dataset_kwargs": {
                "data_path": "../data/celeba64",
                "image_channel": 3,
                "image_size": 64,
                "augmentation": False, 
                "split": "test",
            },
        })
    elif dataset == "ffhq":
        config.update({
            "dataset_name": "FFHQ",
            "dataset_kwargs": {
                "data_path": "../data/ffhq128",
                "image_channel": 3,
                "image_size": 128,
                "augmentation": False, 
                "split": "train",
            },
        })
    
    # determine masks
    model_config = load_yaml(config["config_path"])
    if 'ssl_config' in model_config.keys() and ('dims_per_stage' in model_config['ssl_config'].keys()):
        mask_tuple = ueq_masks(k, latend_dim, device, model_config['ssl_config'])
    elif 'ssl_config' not in model_config.keys():
        mask_tuple = uni_masks(k, latend_dim, device)
        a, b = mask_tuple
        mask_tuple = (None, b)
    else:
        mask_tuple = uni_masks(k, latend_dim, device)

    data_tuple = (img1, img2, split)
    if any(e is not None for e in data_tuple):
        assert None not in (img1, img2, split), f"(img1,img2,split)={(img1, img2, split)} should not contain None."
        # update default config dict
        config["image_index_1"], config["image_index_2"], config["dataset_kwargs"]["split"] = data_tuple

    with context_timer():
        interp = Interp(config, alpha_range, device, mask_tuple, f=f, r=r)
        merge = interp.exp_lerp_simple()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = f"{output_dir}/{method_name}_{config['dataset_kwargs']['split']}_{config['image_index_1']}_{config['image_index_2']}.jpg"
    merge.save(image_path)
    print(f"image saved to {image_path}")

if __name__ == '__main__':
    fire.Fire(main)