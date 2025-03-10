# ICLR 2024: Exploring Diffusion Time-steps for Unsupervised Representation Learning (DiTi)

Official PyTorch implementation of our [ICLR 2024 paper](https://arxiv.org/pdf/2401.11430.pdf). The code can be used to learn image representation in an unsupervised fashion on CelebA, FFHQ or Bedroom dataset. The learned representation enables counterfactual generation and manipulation. This repository is adapted from the PyTorch implementation of [PDAE](https://arxiv.org/abs/2212.12990).
```
@inproceedings{yue2024exploring,
  title={Exploring Diffusion Time-steps for Unsupervised Representation Learning},
  author={Yue, Zhongqi and Wang, Jiankun and Sun, Qianru and Ji, Lei and Chang, Eric I and Zhang, Hanwang},
  booktitle={ICLR},
  year={2024}
}
```

## Dataset

We use the LMDB ready-to-use datasets provided by Diff-AE ([https://github.com/phizaz/diffae#lmdb-datasets](https://github.com/phizaz/diffae#lmdb-datasets)).

LFW dataset is automatically downloaded when running eval/attribute_regression.py. The LFW attributes file need to be manually downloaded at [this link](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt) and move to the location specified below.

The directory structure should be:

```
data
├─ffhq
|  ├─data.mdb
|  └lock.mdb
├─celeba64
|    ├─data.mdb
|    └lock.mdb
├─bedroom
|    ├─data.mdb
|    └lock.mdb
├─lfw
|    ├─lfw-py
|    |     ├─lfw_funneled
|    |     ├─lfw_attributes.txt
|    |     └...
|    └lfw.bin
```




## Download

[pre-trained-dpms](https://drive.google.com/drive/folders/1mU6zgo8WYjNmUtLXZAcsXzv8RghWN9zv?usp=share_link)

Download the models and put them into the pre-trained-dpms directory.

Download DejaVuSans.ttf font file and put into "/$SOMEDIR/miscs/Fonts/dejavu-sans/ttf/DejaVuSans.ttf". Replace the following line with the saved directory in eval/src/eval_utils.py:

```
font = ImageFont.truetype("/$SOMEDIR/miscs/Fonts/dejavu-sans/ttf/DejaVuSans.ttf", 10)
```

## Recommended Project File Structure

We recommend that data/pre-trained-dpms/runs directory are put into the project directory as shown below. The trained models and evaluation results will be saved into ./runs.

```
DiTi
├─config
├─data
├─dataset
├─eval
├─model
├─pre-trained-dpms
├─runs
├─sampler
├─trainer
├─utils
├─.gitignore
└─README.md
```

## Environment

We used PyTorch 2.0 with CUDA driver cu117 for our experiments.


## Training

PDAE training on CelebA with 1 node (n=1) and 2 GPUs (g=2). "-nr=0" specifies the index of the current node (for training with multiple nodes).

```
cd ./trainer
python train_representation_learning.py --config_path=../config/celeba64/pdae.yml --run_path=../runs --method=pdae -n=1 -g=2 -nr=0
```


DiTi training (k=64) on CelebA with 1 node (n=1) and 2 GPUs (g=2). We include the configurations for each dataset in ./config directory.

```
cd ./trainer
python train_representation_learning.py --config_path=../config/celeba64/diti64.yml --run_path=../runs --method=diti -n=1 -g=2 -nr=0
```

You can change the config file and run path in the python file.



## Evaluation

Attribute classification on CelebA with a trained model:

```
cd ./eval
python multitask_classification.py --ckpt_root=../runs/celeba64/diti64 --size=64 --ckpt_name=latest --epochs=15 --device=0 --ema
```

Attribute regression on LFW with a trained model:

```
cd ./eval
python multitask_classification.py --ckpt_root=../runs/celeba64/diti64 --size=64 --ckpt_name=latest --epochs=15 --device=0 --ema
```

Counterfactual generation between 2 images with a trained model. dataset can be celeba or ffhq. Generated images will be saved to ../runs/cf_generations

```
cd ./eval
python counterfactual_generation.py --split train --img1 32972 --img2 32973 --gpu 1 --ckpt_root ../runs/celeba64/pdae --ckpt_name latest --dataset celeba
```

Modular manipulation on a CelebA attribute (e.g., index 25) by setting ProbMask dimension d' to 16. Learned classifier and logging will be saved to the ckpt_root.

```
cd ./eval
python manipulation.py --ckpt_root=../runs/celeba64/diti --ckpt_name=latest --device=0 --ema --promask=16 --epochs=10 --attr_idx=25
```

## Examples

Please refer to notebooks folder for more hands-on examples on interpolation and manipulation.
