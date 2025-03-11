import argparse
import os

import numpy as np
import torch as th
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
    sample_dir = Path(workdir)/f'sample_{step}/interpolation_{args.sampler}_{args.rho}_{args.steps}_w={args.guidance}_churn={args.churn_step_ratio}'
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

    logger.log("Interpolation...")
    img1 = Image.open("assets/img1.png")
    img2 = Image.open("assets/img2.png")


    transform = transforms.Compose([transforms.ToTensor(),])
    img1 = transform(img1)
    img2 = transform(img2)
    imgs = th.stack([img1,img2])

    x0 = imgs.to(dist_util.dev()) * 2 -1
    z = model.encode(x0)

    z_interpol = []
    for i in range(11):
        z_interpol.append(z[0] * 0.1 * (10-i) + z[1] * 0.1 * (i))
    z_interpol = th.stack(z_interpol)

    xT = model.decode(z_interpol)

    model_kwargs = {'xT': xT}
    sample, path, nfe = karras_sample(
        diffusion,
        model,
        xT,
        z_interpol,
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
    vutils.save_image(sample.permute(0,3,1,2).float(), f'{sample_dir}/interpolation_x0.png', normalize=True, nrow=11)

    ## X_T save
    one_path = path[0]
    one_path = ((one_path + 1) * 127.5).clamp(0, 255).to(th.uint8)
    one_path = one_path.permute(0, 2, 3, 1)
    one_path = one_path.contiguous()
    vutils.save_image(one_path.permute(0, 3, 1, 2).float(), f'{sample_dir}/interpolation_xT.png',normalize=True, nrow=11)


    dist.barrier()
    logger.log("Interpolation complete")


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
