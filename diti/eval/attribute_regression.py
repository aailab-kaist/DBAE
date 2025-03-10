import sys
sys.path.append("../")

import argparse
import os
from utils import load_yaml, init_process
import model.representation.encoder as encoder_module

import torch
import torch.nn as nn

from torch.utils.data import Subset, DataLoader
from torch.optim import SGD, Adam
from eval.src.eval_utils import eval_regression
from torch.nn import MSELoss
from src.lfw_attribute import LFWAttribute
from torchvision import transforms
from tqdm import tqdm


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_root', type=str, default=os.path.join("../runs", "celeba64_baseline"))
parser.add_argument('--dataset_root', type=str, default=os.path.join("../data", "lfw"))
parser.add_argument('--ckpt_name', type=str, default="latest", help="name without .pt")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--ema', action="store_true", help="load ema weights for the encoder")
args = parser.parse_args()

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

# constructing dataloaders
train_transform = test_transform = transforms.Compose([
    transforms.Resize(int(args.size * 1.1)),
    transforms.CenterCrop(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,), inplace=True)
])
# default funnel set is already loose crop
train_set = LFWAttribute(args.dataset_root, split='train', transform=train_transform, download=True)
test_set = LFWAttribute(args.dataset_root, split='test', transform=test_transform, download=True)

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
encoder = getattr(encoder_module, config["encoder_config"]["model"], None)(**config["encoder_config"])
encoder.load_state_dict(ckpt['ema_encoder'] if args.ema else ckpt['encoder'])
encoder = encoder.to(device).eval()
encoder.requires_grad_(requires_grad=False)
classifier = nn.Linear(encoder.latent_dim, train_set.num_attributes).to(device)
optimizer = Adam(classifier.parameters(), lr=0.001)

# define loss function
loss_fn = MSELoss()

# prepare logging directory
log_dir_name = "regression"
if args.ema:
    log_dir_name += "_ema"
log_dir = os.path.join(args.ckpt_root, log_dir_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# training
best_r = 0.0
step = 0
test_results = []
test_mse = []
for epoch in range(args.epochs):
    classifier.train()
    encoder.eval()
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        imgs = batch[0].to(device)
        labels = batch[2].to(device).float()
        preds = classifier(encoder(imgs))
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        # writer.add_scalar("multitask/loss", loss, step)
        pbar.set_description(f"Step {step} loss {loss:.3f}")
        step += 1
    pearson_r, mse_per_attribute = eval_regression(test_loader, encoder, classifier, device)
    avg_r = sum(pearson_r)/len(pearson_r)
    avg_mse = mse_per_attribute.mean()
    test_results.append(pearson_r)
    test_mse.append(mse_per_attribute)
    print(f"Epoch {epoch} test avg pearson r: {avg_r:.3f}; avg MSE: {avg_mse:.3f}")
    if avg_r > best_r:
        print(f"New best @Epoch {epoch} with val ap: {avg_r:.3f}")
        best_r = avg_r
        torch.save(classifier.state_dict(), os.path.join(log_dir, "best.pt"))

# Write results
with open(os.path.join(log_dir, "test.txt"), 'a') as file:
    for epoch, result in enumerate(test_results):
        file.write(f"Test pearson r @Epoch{epoch}: {sum(result) / len(result):.3f}.\n")
        file.write(', '.join([str(r) for r in result]))
        file.write(f"Test MSE @Epoch{epoch}: {test_mse[epoch].mean()}.\n")
        file.write(', '.join([str(mse) for mse in test_mse[epoch]]))
        file.write('\n\n')