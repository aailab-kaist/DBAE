"""evaluating multitask performance of encoders on celeba dataset. Supports
- since celeba Dataset class crop celeba images to 128x128 first, encoder's input size is not recommended to be larger than 128.
"""

import sys
sys.path.append("../")

import argparse
import os
from utils import load_yaml, init_process
import model.representation.encoder as encoder_module
import dataset as dataset_module
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torch.optim import SGD, Adam
from eval.src.eval_utils import MultiTaskLoss, eval_multitask
from tqdm import tqdm
import numpy as np

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_root', type=str, default=os.path.join("../runs", "celeba64_baseline"))
parser.add_argument('--ckpt_name', type=str, default="latest", help="name without .pt")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--ema', action="store_true", help="load ema weights for the encoder")
parser.add_argument('--size', type=int, default=None, help="input image size of the encoder")
args = parser.parse_args()

assert args.size is not None, "Please specify proper input size for your encoder in the argument '--size'."

# load config and ckpt
device = f"cuda:{args.device}"
config_path = os.path.join(args.ckpt_root, "config.yml")
ckpt_path = os.path.join(args.ckpt_root, "checkpoints", f"{args.ckpt_name}.pt")
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
unwanted_prefix = '_orig_mod.'
for state_dict in [ckpt['encoder'], ckpt['ema_encoder'], ckpt['decoder'], ckpt['ema_decoder']]:
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
config = load_yaml(config_path)

# constructing celeba datasets/dataloaders
celeba_train_config = {
    "name": "CELEBA64",
    "data_path": "../data/celeba64",
    "image_size": args.size,
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
encoder = getattr(encoder_module, config["encoder_config"]["model"], None)(**config["encoder_config"])
encoder.load_state_dict(ckpt['ema_encoder'] if args.ema else ckpt['encoder'])
encoder = encoder.to(device).eval()
encoder.requires_grad_(requires_grad=False)
classifier = nn.Linear(encoder.latent_dim, train_dataset.num_attributes).to(device)

optimizer = Adam(classifier.parameters(), lr=0.001)

# define loss function
loss_fn = MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

# prepare logging directory
log_dir_name = "multitask"
if args.ema:
    log_dir_name += "_ema"
log_dir = os.path.join(args.ckpt_root, log_dir_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

# training
best_val_ap = 0.0
step = 0
for epoch in range(args.epochs):
    classifier.train()
    encoder.eval()
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        imgs = batch["net_input"]["x_0"].to(device)
        labels = batch["net_input"]["label"].to(device)
        logits = classifier(encoder(imgs))
        loss = loss_fn.compute(logits, labels, return_dict=False)
        loss.backward()
        optimizer.step()
        # writer.add_scalar("multitask/loss", loss, step)
        step += 1
    val_ap = eval_multitask(val_loader, encoder, classifier, device)
    val_ap = sum(val_ap)/len(val_ap)
    print(f"Epoch {epoch} val ap: {val_ap:.3f}")
    if val_ap > best_val_ap:
        print(f"New best @Epoch {epoch} with val ap: {val_ap:.3f}")
        best_val_ap = val_ap
        torch.save(classifier.state_dict(), os.path.join(log_dir, "best.pt"))

# test
classifier_ckpt = torch.load(os.path.join(log_dir, "best.pt"), map_location=torch.device('cpu'))
classifier = nn.Linear(encoder.latent_dim, train_dataset.num_attributes)
print(classifier.load_state_dict(classifier_ckpt))
classifier = classifier.to(device)
test_ap = eval_multitask(test_loader, encoder, classifier, device)
print(f"Test ap:{sum(test_ap) / len(test_ap):.3f}")
with open(os.path.join(log_dir, "test.txt"), 'a') as file:
    file.write(f"Test ap:{sum(test_ap) / len(test_ap):.3f}.\n")
    file.write(', '.join([str(ap) for ap in test_ap]))
    file.write('\n\n')