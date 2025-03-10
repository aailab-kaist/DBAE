import sys
sys.path.append("../")

import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, SGD
from torch.nn import SyncBatchNorm

from utils import move_to_cuda, save_image, load_yaml
from representation_learning_trainer import RepresentationLearningTrainer
from model.representation.ssl import SimSiam, SimCLR, SwinMAE
from pathlib import Path


class DiTiTrainer(RepresentationLearningTrainer):
    def __init__(self, config, run_path, distributed_meta_info, resume, load, use_torch_compile=False):
        self.ssl_start_step = config["ssl_config"]["ssl_start_step"] if "ssl_start_step" in config["ssl_config"].keys() else 0
        self.weight_decay_updated = float(config["optimizer_config"]["weight_decay"])
        super().__init__(config=config, run_path=run_path, distributed_meta_info=distributed_meta_info, resume=resume, use_torch_compile=use_torch_compile)

        if load:
            assert not resume, "load means loading a pretrained model to train under a different config. resume means continuing original config."
            assert len(load.split('@')) == 2, "load must be specified as NAME@CKPT, e.g., PDAE@save-200k"
            model_name, ckpt_name = load.split('@')
            self.load_pretrain(os.path.join(Path(run_path).parent.absolute(), model_name, 'checkpoints', f"{ckpt_name}.pt"))

        self.reset_loss_log()
        self.additional_data_names = ["pdae_loss", "ssl_loss"]
        self.use_ssl_sampler = "ssl_batch_size" in config["ssl_config"].keys()\
              and (config["dataloader_config"]["batch_size"] != config["ssl_config"]["ssl_batch_size"])
        if self.use_ssl_sampler:
            import dataset as dataset_module
            from torch.utils.data import DataLoader
            from utils import set_worker_seed_builder
            train_dataset_config = self.config["train_dataset_config"]
            dataset_name = train_dataset_config["name"]
            train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
            self.ssl_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            dataloader_config = copy.deepcopy(self.config["dataloader_config"])
            dataloader_config["batch_size"] = config["ssl_config"]["ssl_batch_size"] // self.world_size
            self.ssl_dataloader = DataLoader(
                dataset=train_dataset,
                sampler=self.train_sampler,
                pin_memory=True,
                collate_fn=train_dataset.collate_fn,
                worker_init_fn=set_worker_seed_builder(self.rank),
                persistent_workers=True,
                **dataloader_config
            )
            def build_ssl_infinite_cycle():
                if self.rank == 0:
                    base_seed = [int(time.time())]
                else:
                    base_seed = [None]
                torch.distributed.broadcast_object_list(base_seed, src=0, device=self.device)

                base_seed = base_seed[0]
                print(self.rank, "ssl_data_loader_seed", base_seed)
                while True:
                    base_seed += 1
                    self.ssl_sampler.set_epoch(base_seed)
                    for data in self.ssl_dataloader:
                        yield data
            self.ssl_infinite_cycle = build_ssl_infinite_cycle()

    def get_batch_loss(self, batch):
        self.update_weight_decay()
        # SSL loss
        ssl_batch = next(self.ssl_infinite_cycle) if self.use_ssl_sampler else batch
        if self.config["ssl_config"]["ssl_method"] == "mae":
            ssl_loss, _, _ = self.ssl_model(move_to_cuda(ssl_batch["net_input"]["x_ssl1"]), feature=False)
            pdae_z = self.ssl_model(move_to_cuda(batch["net_input"]["x_0"]), feature=True)
        else:
            z12 = self.encoder(move_to_cuda(torch.cat((ssl_batch["net_input"]["x_ssl1"], ssl_batch["net_input"]["x_ssl2"]))))
            ssl_loss = self.ssl_model(z12)
            pdae_z = None

        if (not "no_pdae" in self.config["ssl_config"].keys()) or (not self.config["ssl_config"]["no_pdae"]):
            # PDAE Loss
            pdae_loss, loss_batch, timesteps = self.gaussian_diffusion.pdae_timestep_loss(
                encoder=self.encoder,
                decoder=self.decoder,
                x_0=move_to_cuda(batch["net_input"]["x_0"]),
                t_to_idx=move_to_cuda(self.t_to_idx),
                masks=move_to_cuda(self.masks),
                k_masks=move_to_cuda(self.k_masks),
                receding_masks=move_to_cuda(self.receding_masks),
                mode=self.pdae_mode,
                pdae_z=pdae_z
            )
            self.update_loss_log(loss_batch, timesteps)
        else:
            pdae_loss = torch.zeros_like(ssl_loss)

        ssl_weight = self.config["ssl_config"]["ssl_weight"] \
            if self.step > self.ssl_start_step else 0.0
        total_loss = pdae_loss + ssl_weight * ssl_loss

        logging_data = {
            "pdae_loss": pdae_loss,
            "ssl_loss": ssl_loss
        }
        return total_loss, logging_data
    
    def reset_loss_log(self):
        self.pdae_loss_log = None

    def update_loss_log(self, losses, timesteps):
        pdae_loss_batch, _ = np.histogram(
            timesteps.cpu().numpy(),
            bins=np.arange(self.config['diffusion_config']['timesteps']),
            weights=losses.mean(dim=(1,2,3)).detach().cpu().numpy(),
            density=False
        )
        if self.pdae_loss_log is None:
            self.pdae_loss_log = pdae_loss_batch
        else:
            self.pdae_loss_log += pdae_loss_batch

    def eval(self):
        if self.pdae_loss_log is not None:
            pdae_loss_logs = self.gather_data(self.pdae_loss_log)
            pdae_loss_log = np.array(pdae_loss_logs).mean(axis=0)
            if self.rank == 0:
                from matplotlib import pyplot as plt
                plt.bar(np.arange(self.config['diffusion_config']['timesteps']-1), pdae_loss_log, width=0.5)
                plt.savefig(os.path.join(self.run_path, 'samples', "loss{}k.png".format(self.step // 1000)))
                plt.clf()
            self.reset_loss_log()
        super().eval()
    
    def update_weight_decay(self):
        optimizer_config = self.config["optimizer_config"]
        ssl_configs = self.config["ssl_config"]
        need_to_update_weight_decay = (not ssl_configs["encoder_opt_sgd"]) \
            and ('encoder_weight_decay' in optimizer_config.keys()) \
            and (float(optimizer_config['encoder_weight_decay']) != float(optimizer_config['weight_decay'])) \
            and (self.config["ssl_config"]["ssl_weight"] != 0.0) \
            and (self.step > self.config["ssl_config"]["ssl_start_step"]) \
            and (self.weight_decay_updated == float(optimizer_config['weight_decay']))
        if need_to_update_weight_decay:
            weight_decay = float(optimizer_config['encoder_weight_decay'])
            param_group = self.optimizer.param_groups[-1]
            assert param_group["name"] == "encoder"
            param_group["weight_decay"] = weight_decay
            self.weight_decay_updated = weight_decay
            
    def _build_model(self):
        super()._build_model()
        latent_dim = self.config["encoder_config"]["latent_dim"]
        
        ssl_name = self.config["ssl_config"]["ssl_method"]
        if ssl_name == "simsiam":
            self.ssl_model = SimSiam(
                in_dim=latent_dim,
                hidden_dim=self.config["ssl_config"]["hidden_dim"],
                bottleneck_dim=self.config["ssl_config"]["bottleneck_dim"],
                out_dim=self.config["ssl_config"]["out_dim"]
            )
        elif ssl_name == "simclr":
            assert "simclr_temperature" in self.config["ssl_config"].keys(), "Temperature must be specified for SimCLR!"
            self.ssl_model = SimCLR(
                in_dim=latent_dim,
                hidden_dim=self.config["ssl_config"]["hidden_dim"],
                out_dim=self.config["ssl_config"]["out_dim"],
                temperature=self.config["ssl_config"]["simclr_temperature"],
            )
        elif ssl_name == "mae":
            from functools import partial
            import torch.nn as nn
            assert "mask_ratio" in self.config["ssl_config"].keys()
            self.ssl_model = SwinMAE(
                norm_pix_loss=False, mask_ratio=self.config["ssl_config"]["mask_ratio"],
                img_size=self.config["train_dataset_config"]["image_size"], patch_size=2, in_chans=3,
                decoder_embed_dim=384,
                depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
                window_size=8, qkv_bias=True, mlp_ratio=4,
                drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            assert False

        self.ssl_model.cuda()
        # convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        self.ssl_model = SyncBatchNorm.convert_sync_batchnorm(self.ssl_model)
        self.ssl_model = DistributedDataParallel(self.ssl_model, device_ids=[self.device])

        # setup masks
        if "pdae_mode" in self.config["ssl_config"].keys():
            self.pdae_mode = self.config["ssl_config"]["pdae_mode"]
        else:
            self.pdae_mode = "base"
        assert self.pdae_mode in ["base", "stop_grad", "dual"], "Unsupported PDAE mode!"

        self.k = self.config["ssl_config"]["k"]
        n_timesteps = self.config['diffusion_config']['timesteps']

        if 'stages' in self.config["ssl_config"].keys():
            stages = self.config["ssl_config"]["stages"].split(',')
            self.stages = [int(stage) for stage in stages]
        else:
            self.stages = None

        self.t_to_idx = torch.zeros(n_timesteps).long()
        if 'k_per_stage' in self.config["ssl_config"].keys():
            assert 'stages' in self.config['ssl_config'].keys()
            k_per_stage = self.config["ssl_config"]["k_per_stage"].split(',')
            self.k_per_stage = [int(k) for k in k_per_stage]
            current_stage = 0
            sum_indices = 0
            for t in range(n_timesteps):
                if t == self.stages[current_stage]:
                    sum_indices += self.k_per_stage[current_stage]
                    current_stage += 1
                current_steps = float(self.stages[current_stage])
                current_k = float(self.k_per_stage[current_stage])
                self.t_to_idx[t] = int(float(t) / current_steps * current_k + sum_indices)
        else:
            for t in range(n_timesteps):
                self.t_to_idx[t] = int(float(t) / (float(n_timesteps) / self.k))
            self.k_per_stage = None

        if 'dims_per_stage' in self.config["ssl_config"].keys():
            dims_per_stage = self.config["ssl_config"]["dims_per_stage"].split(',')
            self.dims_per_stage = [int(k) for k in dims_per_stage]
        else:
            self.dims_per_stage = None

        if self.k == 1:
            self.masks = torch.ones(self.k, latent_dim)
            self.k_masks = torch.ones(self.k, latent_dim)
            self.receding_masks = torch.zeros(self.k, latent_dim)
        else:
            self.masks = torch.zeros(self.k, latent_dim)
            self.k_masks = torch.zeros(self.k, latent_dim)
            self.receding_masks = torch.zeros(self.k, latent_dim)
            for i in range(self.k):
                #current_dim = int(float(latent_dim) / self.k * (i + 1))
                #prev_dim = int(float(latent_dim) / self.k * i)
                current_dim = self.get_mask_end_dim(i, latent_dim, self.t_to_idx, self.dims_per_stage)
                prev_dim = self.get_mask_end_dim(i-1, latent_dim, self.t_to_idx, self.dims_per_stage)
                self.masks[i, 0:current_dim] = 1.0
                self.k_masks[i, prev_dim:current_dim] = 1.0
                self.receding_masks[i, 0:prev_dim] = 1.0

    def get_mask_end_dim(self, idx, latent_dim, t_to_idx, dims_per_stage=None):
        if idx >= self.k:
            assert False, 'max idx value is k-1!'
        if idx < 0:
            return 0
        if dims_per_stage is None:
            return int(float(latent_dim) / self.k * (idx+1))
        else:
            assert self.stages is not None, 'Config file error. No stages found under ssl_config!'
            # calculate how many blocks in total after each stage
            accum_num_blocks = np.zeros(len(self.stages))
            for i in range(self.k):
                start_t = torch.nonzero(t_to_idx==i, as_tuple=True)[0][0].item()
                stage = np.argmax(np.array(self.stages) > start_t)
                accum_num_blocks[stage:] += 1
            start_t = torch.nonzero(t_to_idx==idx, as_tuple=True)[0][0].item()
            stage = np.argmax(np.array(self.stages) > start_t)
            stage_total_dim = self.dims_per_stage[stage]
            stage_num_blocks = accum_num_blocks[stage] - accum_num_blocks[stage-1] if stage>0 else accum_num_blocks[stage]
            if int(accum_num_blocks[stage]) == idx + 1:
                return sum(dims_per_stage[0:stage+1])
            else:
                stage_prev_blocks = idx - int(accum_num_blocks[stage-1]) if stage>0 else idx
                return sum(dims_per_stage[0:stage]) + int(float(stage_total_dim) / float(stage_num_blocks)) * (stage_prev_blocks+1)

    def _build_optimizer_mae(self):
        optimizer_config = self.config["optimizer_config"]
        ssl_configs = self.config["ssl_config"]
        dataloader_config = copy.deepcopy(self.config["dataloader_config"])
        global_batch_size = dataloader_config["batch_size"]
        global_ssl_batch_size = ssl_configs["ssl_batch_size"] if "ssl_batch_size" in ssl_configs.keys() else global_batch_size
        local_batch_size = global_batch_size // self.world_size
        local_ssl_batch_size = global_ssl_batch_size // self.world_size
        base_lr = float(optimizer_config["lr"])  * local_batch_size / 32.0
        sgd_base_lr = float(ssl_configs["lr"])  * local_ssl_batch_size / 32.0
        self.optimizer = Adam(
            [
                # encoder is not updated. because MAE uses its own encoder
                {"params": self.decoder.module.label_emb.parameters()},
                {"params": self.decoder.module.shift_middle_block.parameters()},
                {"params": self.decoder.module.shift_output_blocks.parameters()},
                {"params": self.decoder.module.shift_out.parameters()}
            ],
            lr = base_lr,
            betas = eval(optimizer_config["adam_betas"]),
            eps = float(optimizer_config["adam_eps"]),
            weight_decay= float(optimizer_config["weight_decay"]),
        )
        self.sgd_optimizer = torch.optim.AdamW(
            self.ssl_model.parameters(), lr=sgd_base_lr,
            weight_decay=float(self.config["optimizer_config"]['encoder_weight_decay']),
            betas=(0.9, 0.95)
        )
        from torch.optim.lr_scheduler import LambdaLR
        def get_lr(step, total_steps, lr_max, lr_min):
            """Compute learning rate according to cosine annealing schedule."""
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
        total_steps = self.config['runner_config']['max_images'] // self.config['dataloader_config']['batch_size'] - self.ssl_start_step + 10
        self.sgd_scheduler = LambdaLR(
            self.sgd_optimizer,
            lr_lambda=lambda step: get_lr(step, total_steps, sgd_base_lr, sgd_base_lr / 10.0)
        )
        

    def _build_optimizer(self):
        if self.config["ssl_config"]["ssl_method"] == "mae":
            assert self.config["ssl_config"]["encoder_opt_sgd"], "MAE use own optimizer for the encoder"
            return self._build_optimizer_mae()
        
        optimizer_config = self.config["optimizer_config"]
        ssl_configs = self.config["ssl_config"]

        # lr adjustment
        dataloader_config = copy.deepcopy(self.config["dataloader_config"])
        global_batch_size = dataloader_config["batch_size"]
        global_ssl_batch_size = ssl_configs["ssl_batch_size"] if "ssl_batch_size" in ssl_configs.keys() else global_batch_size
        local_batch_size = global_batch_size // self.world_size
        local_ssl_batch_size = global_ssl_batch_size // self.world_size
        base_lr = float(optimizer_config["lr"])  * local_batch_size / 32.0
        sgd_base_lr = float(ssl_configs["lr"])  * local_ssl_batch_size / 32.0

        adam_param_list = [
            {"params": self.decoder.module.label_emb.parameters()},
            {"params": self.decoder.module.shift_middle_block.parameters()},
            {"params": self.decoder.module.shift_output_blocks.parameters()},
            {"params": self.decoder.module.shift_out.parameters()}
        ]
        sgd_param_list = []

        if ssl_configs["encoder_opt_sgd"]:
            sgd_param_list.append({
                'name': 'encoder',
                'params': self.encoder.parameters(),
                'lr': sgd_base_lr
            })
        else:
            adam_param_list.append(
                {"params": self.encoder.parameters(), 'name': 'encoder'}
            )

        if ssl_configs["ssl_opt_sgd"]:
            predictor_prefix = ('module.predictor', 'predictor')
            sgd_param_list.append({
                'name': 'base',
                'params': [param for name, param in self.ssl_model.named_parameters() if (not name.startswith(predictor_prefix))],
                'lr': sgd_base_lr
            })
            sgd_param_list.append({
                'name': 'predictor',
                'params': [param for name, param in self.ssl_model.named_parameters() if name.startswith(predictor_prefix)],
                'lr': sgd_base_lr
            })
        else:
            adam_param_list.append({"params": self.ssl_model.parameters()})

        self.optimizer = Adam(
            adam_param_list,
            lr = base_lr,
            betas = eval(optimizer_config["adam_betas"]),
            eps = float(optimizer_config["adam_eps"]),
            weight_decay= float(optimizer_config["weight_decay"]),
        )

        if len(sgd_param_list) > 0:
            self.sgd_optimizer = SGD(
                sgd_param_list, lr=sgd_base_lr,
                momentum=0.9, weight_decay=1.0e-6
            )

            # from torch.optim.lr_scheduler import LambdaLR
            # def get_lr(step, total_steps, lr_max, lr_min):
            #     """Compute learning rate according to cosine annealing schedule."""
            #     return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
            total_steps = self.config['runner_config']['max_images'] // self.config['dataloader_config']['batch_size'] - self.ssl_start_step + 10
            # self.sgd_scheduler = LambdaLR(
            #     self.sgd_optimizer,
            #     lr_lambda=lambda step: get_lr(step, total_steps, sgd_base_lr, sgd_base_lr / 10.0)
            # )
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.sgd_scheduler = CosineAnnealingLR(
                self.sgd_optimizer, total_steps, sgd_base_lr / 10
            )
        else:
            self.sgd_optimizer = None

    def get_save_data(self):
        data = {
            'step': self.step,
            'encoder': self.encoder.module.state_dict(),
            'ema_encoder': self.ema_encoder.module.state_dict(),
            'decoder': self.decoder.module.state_dict(),
            'ema_decoder': self.ema_decoder.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'ssl_model': self.ssl_model.state_dict()
        }
        if self.sgd_optimizer is not None:
            data['sgd_optimizer'] = self.sgd_optimizer.state_dict()
            data['sgd_scheduler_lr'] = self.sgd_scheduler.state_dict()
        return data

    def load(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))
        unwanted_prefix = '_orig_mod.'
        for state_dict in [data['encoder'], data['ema_encoder'], data['decoder'], data['ema_decoder']]:
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.step = data['step']
        self.encoder.module.load_state_dict(data['encoder'])
        self.ema_encoder.module.load_state_dict(data['ema_encoder'])
        self.decoder.module.load_state_dict(data['decoder'])
        self.ema_decoder.module.load_state_dict(data['ema_decoder'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scaler.load_state_dict(data['scaler'])
        self.ssl_model.load_state_dict(data['ssl_model'])
        if self.sgd_optimizer is not None:
            self.sgd_optimizer.load_state_dict(data['sgd_optimizer'])
            self.sgd_scheduler.load_state_dict(data['sgd_scheduler_lr'])
        print('rank{}: step, model, optimizer and scaler restored from {}(step {}k).'.format(self.rank, path, self.step // 1000))

    def load_pretrain(self, load_path):
        data = torch.load(load_path, map_location=torch.device('cpu'))
        self.step = data['step']
        self.encoder.module.load_state_dict(data['encoder'])
        self.ema_encoder.module.load_state_dict(data['ema_encoder'])
        self.decoder.module.load_state_dict(data['decoder'])
        self.ema_decoder.module.load_state_dict(data['ema_decoder'])
        try:
            self.optimizer.load_state_dict(data['optimizer'])
            self.optimizer.param_groups[-1]['name'] = 'encoder'
        except Exception as e:
            pass
        self.scaler.load_state_dict(data['scaler'])
        # ssl model not loading
        # ssl optimizer (sgd_optimizer) and its scheduler not loading
        print('rank{}: step, model, optimizer and scaler loaded from pretrain {}(step {}k).'.format(self.rank, load_path, self.step // 1000))
