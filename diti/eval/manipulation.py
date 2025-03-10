import sys
sys.path.append("../")

import argparse
import os
from utils import load_yaml, init_process
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from utils import save_image
import dataset as dataset_module
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torch.optim import SGD, Adam
from eval.src.eval_utils import MultiTaskLoss, eval_multitask, accumulate, get_manipulated_images, ProMask
from tqdm import tqdm
import numpy as np
from model.diffusion import GaussianDiffusion


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_root', type=str, default=os.path.join("../runs", "celeba64_baseline"))
parser.add_argument('--ckpt_name', type=str, default="latest", help="name without .pt")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--manip_scale', type=float, default=0.3, help="for visualization only. does not impact the trained classifier")
parser.add_argument('--ema', action="store_true", help="load ema weights for the encoder")
parser.add_argument('--promask', type=int, default=0, help="promask dimension. <=0 means not using promask.")
parser.add_argument('--attr_idx', type=int, default=-1, help="manipulate a specific attribute. <0 means train a classifier for all attributes")
args = parser.parse_args()

# load config and ckpt
device = f"cuda:{args.device}"
config_path = os.path.join(args.ckpt_root, "config.yml")
ckpt_path = os.path.join(args.ckpt_root, "checkpoints", f"{args.ckpt_name}.pt")
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
config = load_yaml(config_path)

# constructing dataloaders
train_dataset_config = config["train_dataset_config"]
val_dataset_config = copy.deepcopy(train_dataset_config)
val_dataset_config.update(config["eval_dataset_config"])
test_dataset_config = copy.deepcopy(train_dataset_config)
test_dataset_config.update({
    'augmentation': False,
    'split': 'test'
})
dataset_name = train_dataset_config["name"]
dataloader_config = config["dataloader_config"]
train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
val_dataset = getattr(dataset_module, dataset_name, None)(val_dataset_config)
test_dataset = getattr(dataset_module, dataset_name, None)(test_dataset_config)

train_loader = DataLoader(
    dataset=train_dataset,
    pin_memory=True,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    batch_size=128,
    num_workers=8,
)
val_loader = DataLoader(
    dataset=val_dataset,
    pin_memory=False,
    collate_fn=val_dataset.collate_fn,
    num_workers=0,
    batch_size=16,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    pin_memory=False,
    collate_fn=test_dataset.collate_fn,
    num_workers=0,
    batch_size=64
)

# constructing models and optimizers
encoder = getattr(encoder_module, config["encoder_config"]["model"], None)(**config["encoder_config"])
encoder.load_state_dict(ckpt['ema_encoder'] if args.ema else ckpt['encoder'])
encoder = encoder.to(device).eval()
encoder.requires_grad_(requires_grad=False)
trained_ddpm_config = load_yaml(config["trained_ddpm_config"])
decoder = getattr(decoder_module, config["decoder_config"]["model"], None)(
    latent_dim=config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"]
)
decoder.load_state_dict(ckpt['ema_decoder'] if args.ema else ckpt['decoder'])
decoder = decoder.to(device).eval()
decoder.requires_grad_(requires_grad=False)
out_dim = train_dataset.num_attributes if args.attr_idx < 0 else 1
classifier_tmp = nn.Linear(encoder.latent_dim, out_dim)
classifier = copy.deepcopy(classifier_tmp).to(device)
ema_classifier = copy.deepcopy(classifier_tmp).to(device)
ema_classifier.eval()
ema_classifier.requires_grad_(False)
del classifier_tmp
if args.promask > 0:
    mask = ProMask(
        encoder.latent_dim, float(args.promask) / encoder.latent_dim,
        0.1, 0.4, args.epochs * len(train_loader) + 1
    ).to(device)
else:
    mask = None
# optimizer = SGD(classifier.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
params = list(classifier.parameters())
if mask:
    params += list(mask.parameters())
optimizer = Adam(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

# construct Gaussian Diffusion
gaussian_diffusion = GaussianDiffusion(config["diffusion_config"], device=device)

# define loss function
multitask = args.attr_idx < 0
if not multitask:
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
else:
    loss_fn = MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

# prepare logging directory
attr_str = train_dataset.id_to_label[args.attr_idx] if not multitask else "all"
mask_dim = args.promask if args.promask > 0 else encoder.latent_dim
log_dir_name = f"mp-{attr_str}-{mask_dim}"
if args.ema:
    log_dir_name += "-ema"
log_dir = os.path.join(args.ckpt_root, log_dir_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(os.path.join(log_dir, 'samples')):
    os.makedirs(os.path.join(log_dir, 'samples'))
writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

# 1. infer latent
z_list = []
if os.path.exists(os.path.join(args.ckpt_root, "stats.pt")):
    data = torch.load(os.path.join(args.ckpt_root, "stats.pt"))
    mean = data["mean"]
    std = data["std"]
else:
    with torch.inference_mode():
        for i, batch in enumerate(train_loader):
            print(i)
            x_0 = batch["net_input"]["x_0"].to(device)
            z = encoder(x_0)
            z_list.append(z.cpu())

        latent = torch.cat(z_list,dim=0)
        mean = latent.mean(0)
        std = latent.std(0)
        torch.save({"mean": mean, "std":std}, os.path.join(args.ckpt_root, "stats.pt"))
mean = mean.to(device)
std = std.to(device)

# 2. train classifier
step = 0
best_val_ap = 0
for epoch in range(args.epochs):
    classifier.train()
    encoder.eval()
    if mask:
        mask.train()
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        imgs = batch["net_input"]["x_0"].to(device)
        labels = batch["net_input"]["label"].to(device)
        with torch.no_grad():
            features = encoder(imgs)
            features = (features - mean) / std
        logits = classifier(mask(features)) if mask else classifier(features)
        if not multitask:
            labels = labels[:, args.attr_idx]
            loss = loss_fn(logits, labels.unsqueeze(1).float())
        else:
            loss = loss_fn.compute(logits, labels, return_dict=False)
        loss.backward()
        if mask:
            torch.nn.utils.clip_grad_norm_(mask.parameters(), 3)
        optimizer.step()
        step += 1
        accumulate(classifier, ema_classifier, decay=0.999)
        if mask:
            mask.adjust_promask(step)
            mask.constrain_mask()
        writer.add_scalar("manipulation/loss", loss, step)
        if mask:
            writer.add_scalar("manipulation/mask_sum", mask.mask_sum(), step)
    if mask:
        mask.eval()
        mask.fix_subnet()
    # eval ap
    val_ap = eval_multitask(
        test_loader, encoder, classifier, device,
        mean, std, args.attr_idx, avg=True, mask=mask
    )
    writer.add_scalar("manipulation/val_ap", val_ap, epoch)
    print(f"Epoch {epoch} val ap: {val_ap:.3f}")
    model_dict = {
        "classifier": classifier.state_dict(),
        "mask": mask.state_dict(),
        "mask_subnet": mask.subnet
    }
    torch.save(model_dict, os.path.join(log_dir, f"{epoch}.pt"))
    if val_ap > best_val_ap:
        print(f"New best @Epoch {epoch} with val ap: {val_ap:.3f}")
        best_val_ap = val_ap

    # plot images
    if epoch % 3 == 0 and epoch != 0:
        manipulated_images, gts = get_manipulated_images(
            encoder, decoder, classifier, gaussian_diffusion, val_loader, mean, std, args.attr_idx, mask, args.manip_scale
        )
        data = {
            'images': manipulated_images.tolist(),
            'gts': gts.tolist()
        }
        manipulated_images = np.asarray(manipulated_images, dtype=np.uint8)
        gts = np.asarray(gts, dtype=np.uint8)
        figure = save_image(manipulated_images, os.path.join(log_dir, 'samples', "sample{}.png".format(epoch)),gts=gts)
        # only writer of rank0 is None
        writer.add_figure("result", figure, epoch)