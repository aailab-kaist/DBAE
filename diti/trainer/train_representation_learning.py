import sys
sys.path.append("../")

import argparse
import os

import torch
import torch.multiprocessing as mp

from representation_learning_trainer import RepresentationLearningTrainer
from trainer.diti_trainer import DiTiTrainer
from utils import load_yaml, init_process
from pathlib import Path

def run(rank, args, distributed_meta_info):
    distributed_meta_info["rank"] = args.nr * args.gpus + rank
    distributed_meta_info["local_rank"] = rank
    init_process(
        init_method=distributed_meta_info["init_method"],
        rank=distributed_meta_info["rank"],
        world_size=distributed_meta_info["world_size"]
    )

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.run_path = os.path.join(args.run_path, os.path.normpath(args.config_path).split('/')[-2], Path(args.config_path).stem)
    restore = os.path.exists(os.path.join(os.path.join(args.run_path, 'checkpoints'), 'latest.pt')) and args.resume
    config = load_yaml(os.path.join(args.run_path, "config.yml")) if restore else load_yaml(args.config_path)
    
    if args.method == "pdae":
        runner = RepresentationLearningTrainer(
            config=config,
            run_path=args.run_path,
            distributed_meta_info=distributed_meta_info,
            resume=args.resume,
            use_torch_compile = args.compile
        )
    elif args.method == "diti":
        runner = DiTiTrainer(
            config=config,
            run_path=args.run_path,
            distributed_meta_info=distributed_meta_info,
            resume=args.resume,
            load=args.load,
            use_torch_compile = args.compile
        )
    else:
        assert False

    runner.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=os.path.join("../config", "ffhq_representation_learning.yml"))
    # Automatically detect run_path/checkpoints/latest.pt to decide whether
    # to restore or
    # to train from scratch using config.yml above.
    parser.add_argument('--run_path', type=str, default=os.path.join("../runs", "ffhq_representation_learning"))
    parser.add_argument('--load', type=str, default="")
    # parser.add_argument('--world_size', type=str, required=True)
    parser.add_argument('--master_addr', type=str, default="127.0.0.1")
    parser.add_argument('--master_port', type=str, default="6666")
    parser.add_argument('--method', type=str, default="pdae", help="pdae | diti")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--compile', action='store_true', help="Turn on to reduce GPU memory usage.")
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
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
        nprocs=args.gpus,
        join=True,
        daemon=False
    )

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_representation_learning.py --world_size 4
