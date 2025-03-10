import sys
sys.path.append("../")
import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as dataset_module
import model.representation.encoder as encoder_module
from utils import load_yaml, move_to_cuda


def main(
        config_path: str,
        checkpoint_path: str,
        gpu: int,
        save_root: str = "../runs/latents"
):
    """infer mean 

    Args:
        config_path (str): .yml config file of PDAE model
            example: ../runs/celeba64/pdae64/config.yml
        checkpoint_path (str): saved ckp of PDAE model
            example: ../runs/celeba64/pdae64/checkpoints/latest.pt
        gpu (int): gpu index
        save_root (str): root path to save inferred latents
    """
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)

    config = {
        "dataset_name": "CELEBA64",
        "data_path": "../data/celeba64",
        "image_channel": 3,
        "image_size": 64,
        "augmentation": False,
    }

    model_config = load_yaml(config_path)
    encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['ema_encoder'])

    dataset_name = config["dataset_name"]
    data_path = config["data_path"]
    image_size = config["image_size"]
    image_channel = config["image_channel"]
    augmentation = config["augmentation"]
    dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel, "augmentation": augmentation, "split": "train"})
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0, batch_size=1000)

    z_list = []

    encoder.cuda().eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            x_0 = move_to_cuda(batch["net_input"]["x_0"])
            z = encoder(x_0)
            z_list.append(z.cpu())

        latent = torch.cat(z_list,dim=0)
        dataset_name = dataset_name.lower()
        model_name = config_path.split("/")[-2]  # in our convention, this place should be the model name
        save_path = os.path.join(save_root, dataset_name + "_" + model_name + ".pt")
        torch.save({"mean": latent.mean(0), "std":latent.std(0)}, save_path)
        print("Latent mean/var. saved to ", save_path)

if __name__ == '__main__':
    fire.Fire(main)