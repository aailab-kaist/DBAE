import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
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
from torch.nn import MSELoss
from ddbm.karras_diffusion import karras_sample
from datasets import load_data
from pathlib import Path
from PIL import Image


import copy
import diti.dataset as dataset_module
from diti.eval.src.eval_utils import eval_regression
from diti.eval.src.lfw_attribute import LFWAttribute
from tqdm import tqdm

from torch.utils.data import Subset, DataLoader
from torch.optim import SGD, Adam
from torchvision import transforms


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
    z_dir = Path(workdir) / f'sample_{step}/downstream_inference'
    dist_util.setup_dist()
    if dist.get_rank() == 0:
        z_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=workdir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model = model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("Eval z...")

    dataset_root = os.path.join("../data", "lfw")
    device = f"cuda:0"
    epochs = 15
    size = args.image_size
    ema = True

    # constructing dataloaders
    train_transform = test_transform = transforms.Compose([
        transforms.Resize(int(size * 1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    # default funnel set is already loose crop
    train_set = LFWAttribute(dataset_root, split='train', transform=train_transform, download=True)
    test_set = LFWAttribute(dataset_root, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(
        dataset=train_set,
        pin_memory=True,
        num_workers=3,
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_set,
        pin_memory=False,
        num_workers=0,
        batch_size=64
    )

    # constructing models and optimizers
    classifier = nn.Linear(512, train_set.num_attributes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.001)

    # define loss function
    loss_fn = MSELoss()

    log_dir_name = "regression"
    if ema:
        log_dir_name += "_ema"
    log_dir = os.path.join(z_dir, log_dir_name)
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # training
    best_r = 0.0
    step = 0
    test_results = []
    test_mse = []
    for epoch in range(epochs):
        classifier.train()
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            imgs = batch[0].to(device)
            imgs = imgs * 2 - 1
            labels = batch[2].to(device).float()
            preds = classifier(model.encode(imgs))
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar("multitask/loss", loss, step)
            pbar.set_description(f"Step {step} loss {loss:.3f}")
            step += 1
        pearson_r, mse_per_attribute = eval_regression(test_loader, model, classifier, device)
        avg_r = sum(pearson_r) / len(pearson_r)
        avg_mse = mse_per_attribute.mean()
        test_results.append(pearson_r)
        test_mse.append(mse_per_attribute)
        print(f"Epoch {epoch} test avg pearson r: {avg_r:.3f}; avg MSE: {avg_mse:.3f}")
        if avg_r > best_r:
            print(f"New best @Epoch {epoch} with val ap: {avg_r:.3f}")
            best_r = avg_r
            th.save(classifier.state_dict(), os.path.join(log_dir, "best.pt"))

    # Write results
    with open(os.path.join(log_dir, "test.txt"), 'a') as file:
        for epoch, result in enumerate(test_results):
            file.write(f"Test pearson r @Epoch{epoch}: {sum(result) / len(result):.3f}.\n")
            file.write(', '.join([str(r) for r in result]))
            file.write(f"Test MSE @Epoch{epoch}: {test_mse[epoch].mean()}.\n")
            file.write(', '.join([str(mse) for mse in test_mse[epoch]]))
            file.write('\n\n')

    dist.barrier()
    logger.log("z_eval complete")

    


        




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
