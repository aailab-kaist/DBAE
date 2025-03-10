import sys
sys.path.append("../")

import argparse
from pathlib import Path
import os

import torch
import torch.multiprocessing as mp

from manipulation_diffusion_trainer import ManipulationDiffusionTrainer
from promask_manipulation_trainer import ProMaskManipulationTrainer
from utils import load_yaml, init_process

celeba_id_to_label = [
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

def run(rank, args, distributed_meta_info):
    distributed_meta_info["rank"] = rank
    init_process(
        init_method=distributed_meta_info["init_method"],
        rank=distributed_meta_info["rank"],
        world_size=distributed_meta_info["world_size"]
    )

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.run_path = os.path.join(args.run_path, os.path.normpath(args.config_path).split('/')[-2], Path(args.config_path).stem)
    restore = os.path.exists(os.path.join(os.path.join(args.run_path, 'checkpoints'), 'latest.pt'))
    config = load_yaml(os.path.join(args.run_path, "config.yml")) if restore else load_yaml(args.config_path)

    # run-time updating configs by command-line args
    # for manipulating a single attribute
    if args.target_index is not None:
        config["runner_config"]["target_index"] = args.target_index
    # if dataset=CELEBA64, add attribute-specific suffix to run_path
    if config["runner_config"]["target_index"] is not None and config["train_dataset_config"]["name"]=="CELEBA64":
        i = config["runner_config"]["target_index"]
        args.run_path += f"_{i}_{celeba_id_to_label[i]}"

    if args.method == "pdae":
        runner = ManipulationDiffusionTrainer(
            config=config,
            run_path=args.run_path,
            distributed_meta_info=distributed_meta_info
        )
    elif args.method == "promask":
        runner = ProMaskManipulationTrainer(
            config=config,
            run_path=args.run_path,
            distributed_meta_info=distributed_meta_info
        )
    else:
        raise NotImplementedError

    runner.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=os.path.join("../config", "celebahq_manipulation.yml"))
    # Automatically detect run_path/checkpoints/latest.pt to decide whether
    # to restore or
    # to train from scratch using config.yml above.
    parser.add_argument('--run_path', type=str, default=os.path.join("../runs", "celebahq_manipulation"))
    parser.add_argument('--world_size', type=str, required=True)
    parser.add_argument('--master_addr', type=str, default="127.0.0.1")
    parser.add_argument('--master_port', type=str, default="6666")
    parser.add_argument('--method', type=str, default="pdae", help="pdae | promask")
    parser.add_argument('-t', '--target_index', type=int, default=None, help="target attribute id to train the classifier and manipulate")

    args = parser.parse_args()

    world_size = int(args.world_size)
    init_method = "tcp://{}:{}".format(args.master_addr, args.master_port)

    distributed_meta_info = {
        "world_size": world_size,
        "master_addr": args.master_addr,
        "init_method": init_method,
        # rank will be added in spawned processes
    }

    mp.spawn(
        fn=run,
        args=(args, distributed_meta_info),
        nprocs=world_size,
        join=True,
        daemon=False
    )

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_manipulation_diffusion.py --world_size 4
