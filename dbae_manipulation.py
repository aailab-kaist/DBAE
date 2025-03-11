import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils
import torch.distributed as dist
from torchvision import transforms

from ddbm import dist_util, logger
from ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from ddbm.karras_diffusion import karras_sample

from datasets import load_data

from pathlib import Path
import math
from PIL import Image
def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def main():
    args = create_argparser().parse_args()
    print(args)

    workdir = os.path.dirname(args.model_path)

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir)/f'sample_{step}/manipulation_{args.sampler}_{args.rho}_{args.steps}_w={args.guidance}_churn={args.churn_step_ratio}'
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Manipulation...")
    classifier = nn.Linear(512, 40)
    classifier.load_state_dict(th.load("ckpt/ffhq/sto/best.pt"))
    classifier = classifier.to(dist_util.dev())
    classifier.eval()

    ## Example 1
    img3 = Image.open("assets/img3.png")
    transform = transforms.Compose([transforms.ToTensor(),])
    img3 = transform(img3)
    img3 = img3.unsqueeze(0)
    change_index = 20 ## You can change attribute index
    fix_xT = True
    x0 = img3.to(dist_util.dev()) * 2 - 1
    z = model.encode(x0)
    scale = th.tensor([[-0.3], [-0.15], [0.], [0.15], [0.3]]).to(dist_util.dev())

    z_norm_manipulated = z + scale * math.sqrt(512) * F.normalize(classifier.weight[change_index][None, :], dim=1)

    if fix_xT:
        xT = model.decode(z)
        xT = xT.repeat(5,1,1,1)
    else:
        xT = model.decode(z_norm_manipulated)


    model_kwargs = {'xT': xT}
    sample, path, nfe = karras_sample(
        diffusion,
        model,
        xT,
        z_norm_manipulated,
        x0,
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=diffusion.sigma_min,
        sigma_max=diffusion.sigma_max,
        churn_step_ratio=args.churn_step_ratio,
        rho=args.rho,
        guidance=args.guidance,
    )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    vutils.save_image(sample.permute(0,3,1,2).float(), f'{sample_dir}/img3_manipulated_x0.png', normalize=True, nrow=5)

    ## X_T save
    one_path = path[0]
    one_path = ((one_path + 1) * 127.5).clamp(0, 255).to(th.uint8)
    one_path = one_path.permute(0, 2, 3, 1)
    one_path = one_path.contiguous()
    vutils.save_image(one_path.permute(0, 3, 1, 2).float(), f'{sample_dir}/img3_manipulated_xT.png',normalize=True, nrow=5)


    ## Example 2
    img4 = Image.open("assets/img4.png")
    transform = transforms.Compose([transforms.ToTensor(), ])
    img4 = transform(img4)
    img4 = img4.unsqueeze(0)
    change_index = 39  ## You can change attribute index
    fix_xT = True
    x0 = img4.to(dist_util.dev()) * 2 - 1
    z = model.encode(x0)
    scale = th.tensor([[-0.2], [-0.1], [0.], [0.1], [0.2]]).to(dist_util.dev())

    z_norm_manipulated = z + scale * math.sqrt(512) * F.normalize(classifier.weight[change_index][None, :], dim=1)

    if fix_xT:
        xT = model.decode(z)
        xT = xT.repeat(5, 1, 1, 1)
    else:
        xT = model.decode(z_norm_manipulated)

    model_kwargs = {'xT': xT}
    sample, path, nfe = karras_sample(
        diffusion,
        model,
        xT,
        z_norm_manipulated,
        x0,
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=diffusion.sigma_min,
        sigma_max=diffusion.sigma_max,
        churn_step_ratio=args.churn_step_ratio,
        rho=args.rho,
        guidance=args.guidance,
    )

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    vutils.save_image(sample.permute(0, 3, 1, 2).float(), f'{sample_dir}/img4_manipulated_x0.png', normalize=True, nrow=5)

    ## X_T save
    one_path = path[0]
    one_path = ((one_path + 1) * 127.5).clamp(0, 255).to(th.uint8)
    one_path = one_path.permute(0, 2, 3, 1)
    one_path = one_path.contiguous()
    vutils.save_image(one_path.permute(0, 3, 1, 2).float(), f'{sample_dir}/img4_manipulated_xT.png', normalize=True, nrow=5)


    dist.barrier()
    logger.log("Manipulation complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset='ffhq',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='train',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
        latent_dim=512,
        sto=False,
        end=False,
        stoxt=False,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
