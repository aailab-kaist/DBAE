import os
from utils import load_yaml
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
import dataset as dataset_module
import copy

import torch
import torch.nn as nn
from eval.src.eval_utils import ProMask, ImageCanvas
from model.diffusion import GaussianDiffusion
from PIL import Image
id_to_label = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
label_to_id = {v: k for k, v in enumerate(id_to_label)}


def get_manipulated_images(x_0, inferred_x_T, encoder, decoder, classifier, diffusion, mean, std, attr_idx, mask=None, scale=0.3, ddim_style=f'ddim200'):
    if attr_idx < 0:
        attr_idx = 31   # default
    else:
        attr_idx = 0    # if single attribute, then classifier only has 1 weight
    with torch.no_grad():
        weight = mask(classifier.weight) if mask else classifier.weight
        images = diffusion.manipulation_sample(
            classifier_weight=weight,
            encoder=encoder,
            decoder=decoder,
            x_0=x_0,
            inferred_x_T=inferred_x_T,
            latents_mean=mean.to(diffusion.device),
            latents_std=std.to(diffusion.device),
            scale=scale,
            ddim_style=ddim_style,
            verbose = False,
        )
    images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
    images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    return Image.fromarray(images[0])

def manipulate(
    promask = 16,  # 16, 32 or 128
    attr_idx = 36, # Lipstick
    img_idx = 32981,
    epoch = 9,  # 0~9
    manip_scale_range = [-1, -.8, -.6,-.4, -.2, 0, .2, .4, .6, .8, 1],
    gpu = 0,
    ema = True,
    f: int = 1000,
    r: int = 100,
    ckpt_root_pdae: str = 'path/to/ckpt_root_1',
    ckpt_root_diti: str = 'path/to/ckpt_root_2',
    data_path: str = 'path/to/data_path',
    ckpt_name_pdae: str = 'latest',
    ckpt_name_diti: str = 'latest',
):
    # shared
    # useless args during inference
    mask_maxiter = 10 * 162770 / 128 + 1

    img_txt_list = []
    print(f"Target Attribute: {id_to_label[attr_idx]}")
    for ckpt_root, ckpt_name in [(ckpt_root_pdae, ckpt_name_pdae), (ckpt_root_diti, ckpt_name_diti)]:
        # load config and ckpt
        device = f"cuda:{gpu}"
        config_path = os.path.join(ckpt_root, "config.yml")
        ckpt_path = os.path.join(ckpt_root, "checkpoints", f"{ckpt_name}.pt")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        config = load_yaml(config_path)

        # constructing dataloaders
        train_dataset_config = {
            "name": "CELEBA64",
            "data_path": data_path,
            "image_channel": 3,
            "image_size": 64,
            "augmentation": False, 
            "split": "eval",
        }
        dataset_name = train_dataset_config["name"]
        try:
            train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
        except FileNotFoundError:
            train_dataset_config['data_path'] = train_dataset_config['data_path'][3:]
            train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
        # constructing models and optimizers
        encoder = getattr(encoder_module, config["encoder_config"]["model"], None)(**config["encoder_config"])
        encoder.load_state_dict(ckpt['ema_encoder'] if ema else ckpt['encoder'])
        encoder = encoder.to(device).eval()
        encoder.requires_grad_(requires_grad=False)
        try:
            trained_ddpm_config = load_yaml(config["trained_ddpm_config"])
        except FileNotFoundError:
            trained_ddpm_config = load_yaml(config["trained_ddpm_config"][3:])
        decoder = getattr(decoder_module, config["decoder_config"]["model"], None)(
            latent_dim=config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"]
        )
        decoder.load_state_dict(ckpt['ema_decoder'] if ema else ckpt['decoder'])
        decoder = decoder.to(device).eval()
        decoder.requires_grad_(requires_grad=False)
        out_dim = train_dataset.num_attributes if attr_idx < 0 else 1
        classifier_tmp = nn.Linear(encoder.latent_dim, out_dim)
        classifier = copy.deepcopy(classifier_tmp).to(device)
        del classifier_tmp
        if promask > 0:
            mask = ProMask(
                encoder.latent_dim, float(promask) / encoder.latent_dim,
                0.1, 0.4, mask_maxiter
            ).to(device)
        else:
            mask = None
        if mask:
            mask.eval()
            mask.fix_subnet()

        gaussian_diffusion = GaussianDiffusion(config["diffusion_config"], device=device)

        multitask = attr_idx < 0
        # prepare logging directory
        attr_str = train_dataset.id_to_label[attr_idx] if not multitask else "all"
        mask_dim = promask if promask > 0 else encoder.latent_dim
        log_dir_name = f"mp-{attr_str}-{mask_dim}"
        if ema:
            log_dir_name += "-ema"
        log_dir = os.path.join(ckpt_root, log_dir_name)
        
        # load classifier, promask statedict
        manip_ckp = torch.load(os.path.join(log_dir, f"{epoch}.pt"), map_location=torch.device(device))
        classifier.load_state_dict(manip_ckp["classifier"])
        if mask:
            mask.load_state_dict(manip_ckp["mask"])
            mask.subnet = manip_ckp["mask_subnet"]

        # import mean/var latents
        data = torch.load(os.path.join(ckpt_root, "stats.pt"))
        mean = data["mean"].to(device)
        std = data["std"].to(device)

        img = train_dataset.__getitem__(img_idx)
        x_0 = img["x_0"].unsqueeze(0).to(device)
        inferred_x_T = gaussian_diffusion.representation_learning_ddim_encode(f'ddim{f}', encoder, decoder, x_0)

        for manip_scale in manip_scale_range:
            manipulated_images = get_manipulated_images(
                x_0, inferred_x_T, encoder, decoder, classifier, gaussian_diffusion, mean, std, attr_idx, mask, manip_scale, ddim_style=f"ddim{r}"
            )
            img_txt_list.append((manipulated_images, f"{manip_scale:.2f}"))

    nrow = 2
    merge = ImageCanvas.build_from_list(
        img_txt_list, 
        img_txt_list[0][0].size, 
        num_text_image_rows=nrow,
        text_image_height_ratio=0.2,
        txt_shift=0)
    return merge
