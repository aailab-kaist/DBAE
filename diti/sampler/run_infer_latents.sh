#!/bin/bash

CONFIG_PATH="../runs/celeba64/pdae64_ueq_dim/config.yml"
CHECKPOINT_PATH="../runs/celeba64/pdae64_ueq_dim/checkpoints/latest.pt"
GPU=1

python infer_latents.py $CONFIG_PATH $CHECKPOINT_PATH $GPU