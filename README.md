# Diffusion Bridge AutoEncoders for Unsupervised Representation Learning (DBAE) under review in Neurips2024


# Dependencies

Download all the packages described in setup.py
```
pip install -e .
```

# Datasets

1. For FFHQ, CelebA, CelebA-HQ, and LSUN datasets (LMDB format) please follow instructions from https://github.com/phizaz/diffae.
2. Set your dataset path at args.sh.


# Model training 

We provide training bash file train_dbae.sh with dbae_train.py.
Set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` sets dataset. We support FFHQ, CelebA, CelebA-HQ, and LSUNs.
- `SCHEDULE_TYPE` sets forward diffusion. Set `ve` or `vp`. 

To train, run

```
bash train_dbae.sh $DATASET_NAME $SCHEDULE_TYPE
bash train_dbae.sh ffhq vp
```

# Reconstruction

We provide reconstruction bash file recon_dbae.sh with dbae_reconstruction.py.
Set variables `MODEL_PATH`, `CHURN_STEP_RATIO`, `RHO`, `GEN_SAMPLER`, and `N` :
- `MODEL_PATH` sets your checkpoint path
- `CHURN_STEP_RATIO` sets SDE(0~1) or ODE (0) sampling.
- `RHO` sets time-discretization interval seletion.
- `GEN_SAMPLER` sets order of sampler.
- `N` sets sampling step number.

To to reconstruction, run

```
bash recon_dbae.sh $DATASET_NAME $SCHEDULE_TYPE $MODEL_PATH $MODEL_PATH $CHURN_STEP_RATIO 1 train $RHO $GEN_SAMPLER $N
bash recon_dbae.sh ffhq vp {Your_Path}/Neurips2024/DBAE/workdir/0.5_end_sto_k_latent_ffhq128_128_512d_vp/ema_0.9999_1020000.pt 0.0 1 train 7 euler 20
bash recon_dbae.sh ffhq vp {Your_Path}/Neurips2024/DBAE/workdir/0.5_end_sto_k_latent_ffhq128_128_512d_vp/ema_0.9999_1020000.pt 0.33 1 train 7 heun 83
```







This code is heavily built upon the code from followings
- https://github.com/alexzhou907/DDBM
- https://github.com/phizaz/diffae
- https://github.com/odegeasslbc/FastGAN-pytorch

For evaluation we utilize the code from followings (we have plan to combine evaluation module to our own code and release)
- Table1: https://github.com/yue-zhongqi/diti
- Table2: https://github.com/phizaz/diffae
- Table3: https://github.com/GaParmar/clean-fid / https://github.com/ericyeats/nashae-beamsynthesis
- Table4: https://github.com/GaParmar/clean-fid / https://github.com/openai/guided-diffusion
