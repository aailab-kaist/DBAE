import argparse
import os

import numpy as np
import torch as th
import torchvision.utils as vutils
import torch.distributed as dist

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
    sample_dir = Path(workdir)/f'sample_{step}/{args.sampler}_{args.rho}_{args.steps}_w={args.guidance}_churn={args.churn_step_ratio}'
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

    logger.log("sampling...")
    

    all_images = []
    

    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_test=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.split =='train':
        dataloader = all_dataloaders[1]
    elif args.split == 'test':
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError
    args.num_samples = len(dataloader.dataset)

    
    for i, data in enumerate(dataloader):
        
        x0_image = data[0]
        x0 = x0_image.to(dist_util.dev()) * 2 -1

        z = model.encode(x0)
        xT = model.decode(z)

        model_kwargs = {'xT': xT}
        index = data[2].to(dist_util.dev())

        sample, path, nfe = karras_sample(
            diffusion,
            model,
            xT,
            z,
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
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if index is not None:
            gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_index, index)
            gathered_samples = th.cat(gathered_samples)
            gathered_index = th.cat(gathered_index)
            gathered_samples = gathered_samples[th.argsort(gathered_index)]
        else:
            gathered_samples = th.cat(gathered_samples)
        num_display = min(32, sample.shape[0])

        vutils.save_image(sample.permute(0,3,1,2)[:num_display].float(), f'{sample_dir}/{dist.get_rank()}_sample_{i}.png', normalize=True,  nrow=int(np.sqrt(num_display)))
        if x0 is not None:
            vutils.save_image(x0_image[:num_display], f'{sample_dir}/{dist.get_rank()}_x_{i}.png',nrow=int(np.sqrt(num_display)))

        ## X_T save
        one_path = path[0]
        one_path = ((one_path + 1) * 127.5).clamp(0, 255).to(th.uint8)
        one_path = one_path.permute(0, 2, 3, 1)
        one_path = one_path.contiguous()
        vutils.save_image(one_path.permute(0, 3, 1, 2)[:num_display].float(), f'{sample_dir}/x_T_sample_{i}.png',normalize=True, nrow=int(np.sqrt(num_display)))
        all_images.append(gathered_samples.detach().cpu().numpy())

        x0_image = ((x0_image) * 255).clamp(0, 255).to(th.uint8)
        x0_image = x0_image.permute(0, 2, 3, 1)
        x0_image = x0_image.contiguous()

        np.savez(f"{sample_dir}/{dist.get_rank()}_real_sample_{i}.npz", samples=x0_image.numpy())
        np.savez(f"{sample_dir}/{dist.get_rank()}_fake_sample_{i}.npz", samples=sample.to("cpu").numpy())

    logger.log(f"created {len(all_images) * args.batch_size * dist.get_world_size()} samples")
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
    dist.barrier()
    logger.log("sampling complete")


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
