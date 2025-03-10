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

from ddbm.karras_diffusion import karras_sample
from datasets import load_data
from pathlib import Path
from PIL import Image


import copy
import diti.dataset as dataset_module
from diti.eval.src.eval_utils import MultiTaskLoss, eval_multitask
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

    ## Diti arguments
    device = f"cuda:0"
    epochs = 15
    size = args.image_size
    ema = True

    # constructing dataloaders
    celeba_train_config = {
        "name": "CELEBA64",
        "data_path": "/home/aailab/alsdudrla10/SecondArticle/DBAE/PDAE/data/celeba.lmdb",
        "image_size": size,
        "image_channel": 3,
        "latent_dim": 512,
        "augmentation": True,
        "split": "train",
    }
    celeba_val_config = copy.deepcopy(celeba_train_config)
    celeba_val_config.update({
        "augmentation": False,
        "split": "valid",
    })
    celeba_test_config = copy.deepcopy(celeba_train_config)
    celeba_test_config.update({
        'augmentation': False,
        'split': 'test'
    })
    dataset_name = celeba_train_config["name"]
    train_dataset = getattr(dataset_module, dataset_name, None)(celeba_train_config)
    val_dataset = getattr(dataset_module, dataset_name, None)(celeba_val_config)
    test_dataset = getattr(dataset_module, dataset_name, None)(celeba_test_config)

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=3,
        batch_size=128,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
        batch_size=64
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        batch_size=64
    )

    # constructing models and optimizers
    classifier = nn.Linear(512, train_dataset.num_attributes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.001)
    # define loss function
    loss_fn = MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    log_dir_name = "multitask"
    if ema:
        log_dir_name += "_ema"
    log_dir = os.path.join(z_dir, log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # training
    best_val_ap = 0.0
    step = 0
    for epoch in range(epochs):
        classifier.train()
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            imgs = batch["net_input"]["x_0"].to(device)
            imgs = imgs * 2 - 1

            labels = batch["net_input"]["label"].to(device)
            logits = classifier(model.encode(imgs))
            loss = loss_fn.compute(logits, labels, return_dict=False)
            loss.backward()
            optimizer.step()
            # writer.add_scalar("multitask/loss", loss, step)
            step += 1
        val_ap = eval_multitask(val_loader, model, classifier, device)
        val_ap = sum(val_ap) / len(val_ap)
        print(f"Epoch {epoch} val ap: {val_ap:.3f}")
        if val_ap > best_val_ap:
            print(f"New best @Epoch {epoch} with val ap: {val_ap:.3f}")
            best_val_ap = val_ap
            th.save(classifier.state_dict(), os.path.join(log_dir, "best.pt"))

    # test
    classifier_ckpt = th.load(os.path.join(log_dir, "best.pt"), map_location=th.device('cpu'))
    classifier = nn.Linear(512, train_dataset.num_attributes)
    print(classifier.load_state_dict(classifier_ckpt))
    classifier = classifier.to(device)
    test_ap = eval_multitask(test_loader, model, classifier, device)
    print(f"Test ap:{sum(test_ap) / len(test_ap):.3f}")
    with open(os.path.join(log_dir, "test.txt"), 'a') as file:
        file.write(f"Test ap:{sum(test_ap) / len(test_ap):.3f}.\n")
        file.write(', '.join([str(ap) for ap in test_ap]))
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
