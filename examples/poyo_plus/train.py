import copy
import logging
from collections import defaultdict
from typing import Callable, Dict

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.optim import SparseLamb
from torch_brain.models import POYOPlus
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.transforms import Compose
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    MultiTaskDecodingStitchEvaluator,
    DataForMultiTaskDecodingStitchEvaluator,
)


# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        model: POYOPlus,
        cfg: DictConfig,
    ):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        special_emb_params = (
            list(self.model.unit_emb.parameters())
            + list(self.model.session_emb.parameters())
            + list(self.model.readout.parameters())
        )

        remaining_params = [
            p
            for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n and "readout" not in n
        ]

        optimizer = SparseLamb(
            [
                {"params": special_emb_params, "sparse": True},
                {"params": remaining_params},
            ],
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        print("batch_idx", batch_idx)
        print("x_pad2d batch shape:", batch["model_inputs"]["x_pad2d"].shape)
        print("x_pad2d_mask batch shape:", batch["model_inputs"]["x_pad2d_mask"].shape)
    
        # for i in range(3):
        #     mask_false = (batch['model_inputs']['input_mask'][i] == False).nonzero()
        #     print(f"sample {i} padding starts at:", mask_false)
        #     if len(mask_false) > 0:
        #         zero_from = mask_false[0].item()
        #         print(f"sample {i} input_values from padding start:")
        #         print(batch['model_inputs']['input_values'][i][zero_from:])
                
        for i in range(3):
            mask_false = (batch['model_inputs']['input_mask'][i] == False).nonzero()
            print(i, mask_false[0].item() if len(mask_false) > 0 else "no padding")
            if len(mask_false) > 0:
                zero_from = mask_false[0].item()
                tail = batch['model_inputs']['input_values'][i][zero_from:]
                all_zero = (tail == 0).all().item()
                num_zero = (tail == 0).sum().item()
                print(f"sample {i} all zeros from padding start: {all_zero}, num zeros: {num_zero}, tail length: {len(tail)}")

        
        
        import pdb; pdb.set_trace()
# (pdb) batch #! Data -> dict : train_transforms -> model.tokenize()
# dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
# for k in batch.keys(): print(k, batch[k].shape)
# for k in batch['model_inputs'].keys(): print(k, batch['model_inputs'][k].shape)
# input_unit_index torch.Size([3, 528])
# input_timestamps torch.Size([3, 528])
# input_values torch.Size([3, 528, 1])
# input_mask torch.Size([3, 528])
# latent_index torch.Size([3, 128])
# latent_timestamps torch.Size([3, 128])
# output_session_index torch.Size([3, 8])
# output_timestamps torch.Size([3, 8])
# output_decoder_index torch.Size([3, 8])

# p {k: v.shape for k,v in batch['target_values'].items()}
# {'drifting_gratings_temporal_frequency': torch.Size([3]), 'drifting_gratings_orientation': torch.Size([3])}

# {'drifting_gratings_temporal_frequency': tensor([4, 1, 1], device='cuda:0'), 'drifting_gratings_orientation': tensor([3, 4, 2], device='cuda:0')}

# p {k: v.shape for k,v in batch['target_weights'].items()}
# {'drifting_gratings_temporal_frequency': torch.Size([3]), 'drifting_gratings_orientation': torch.Size([3])}
# batch['target_weights']
# {'drifting_gratings_temporal_frequency': tensor([1., 1., 1.], device='cuda:0'), 'drifting_gratings_orientation': tensor([1., 1., 1.], device='cuda:0')}

# batch['session_id']
# ['allen_visual_coding_ophys_2016/710504563', 'allen_visual_coding_ophys_2016/710504563', 'allen_visual_coding_ophys_2016/710504563']
# batch['absolute_start']
# tensor([2018.1108,   53.8092, 1777.1660], device='cuda:0', dtype=torch.float64)
# p {k: v.shape for k,v in batch['eval_mask'].items()}


# {'model_inputs': {'input_unit_index': tensor([[ 1,  2,  3,  ...,  0,  0,  0],
#         [ 1,  2,  3,  ..., 16, 17,  0],
#         [ 1,  2,  3,  ...,  0,  0,  0],
#         ...,
#         [ 1,  2,  3,  ...,  0,  0,  0],
#         [ 1,  2,  3,  ...,  0,  0,  0],
#         [ 1,  2,  3,  ...,  0,  0,  0]], device='cuda:0'), 'input_timestamps': 
# tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000,  ..., 0.9974, 0.9974, 0.0000],
#         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         ...,
#         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],
#        device='cuda:0', dtype=torch.float64), 'input_values': tensor([[[ 
# 0.0702],
#          [ 0.0647],
#          [ 0.0310],
#          ...,
#          [ 0.0000],
#          [ 0.0000],
#          [ 0.0000]],

#         [[ 0.7050],
#          [ 0.0385],
#          [-0.1360],
#          ...,
#          [ 0.0679],
#          [ 0.1751],
#          [ 0.0000]],

#         [[-0.0550],
#          [ 0.1854],
#          [-0.0174],
#          ...,
#          [ 0.0000],
#          [ 0.0000],
#          [ 0.0000]],

#         ...,

#         [[-0.0775],
#          [-0.0627],
#          [-0.0442],
#          ...,
#          [ 0.0000],
#          [ 0.0000],
#          [ 0.0000]],

#         [[ 0.2961],
#          [ 0.0688],
#          [ 0.0883],
#          ...,
#          [ 0.0000],
#          [ 0.0000],
#          [ 0.0000]],

#         [[ 0.2229],
#          [-0.0659],
#          [ 0.1019],
#          ...,
#          [ 0.0000],
#          [ 0.0000],
#          [ 0.0000]]], device='cuda:0'), 'input_mask': tensor([[ True,  True,  
# True,  ..., False, False, False],
#         [ True,  True,  True,  ...,  True,  True, False],
#         [ True,  True,  True,  ..., False, False, False],
#         ...,
#         [ True,  True,  True,  ..., False, False, False],
#         [ True,  True,  True,  ..., False, False, False],
#         [ True,  True,  True,  ..., False, False, False]], device='cuda:0'), 
# 'latent_index': tensor([[ 0,  1,  2,  ..., 13, 14, 15],
#         [ 0,  1,  2,  ..., 13, 14, 15],
#         [ 0,  1,  2,  ..., 13, 14, 15],
#         ...,
#         [ 0,  1,  2,  ..., 13, 14, 15],
#         [ 0,  1,  2,  ..., 13, 14, 15],
#         [ 0,  1,  2,  ..., 13, 14, 15]], device='cuda:0'), 'latent_timestamps':
# tensor([[0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375],
#         [0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375],
#         [0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375],
#         ...,
#         [0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375],
#         [0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375],
#         [0.0625, 0.0625, 0.0625,  ..., 0.9375, 0.9375, 0.9375]],
#        device='cuda:0', dtype=torch.float64), 'output_session_index': 
# tensor([[1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0]], device='cuda:0'), 'output_timestamps': 
# tensor([[0.2129, 0.2129, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9314, 0.9314, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9594, 0.9594, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8832, 0.8832, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8130, 0.8130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3720, 0.3720, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3545, 0.3545, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7124, 0.7124, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6954, 0.6954, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0939, 0.0939, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7212, 0.7212, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0193, 0.0193, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9334, 0.9334, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6842, 0.6842, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0193, 0.0193, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7427, 0.7427, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3578, 0.3578, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8635, 0.8635, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5319, 0.5319, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8473, 0.8473, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2882, 0.2882, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2717, 0.2717, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9807, 0.9807, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8798, 0.8798, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7485, 0.7485, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3407, 0.3407, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0924, 0.0924, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6588, 0.6588, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1774, 0.1774, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9217, 0.9217, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6453, 0.6453, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9044, 0.9044, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3034, 0.3034, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6428, 0.6428, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7863, 0.7863, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8515, 0.8515, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1735, 0.1735, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0561, 0.0561, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9250, 0.9250, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4752, 0.4752, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6809, 0.6809, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0709, 0.0709, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2513, 0.2513, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5225, 0.5225, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2653, 0.2653, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2712, 0.2712, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4668, 0.4668, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6916, 0.6916, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5225, 0.5225, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6952, 0.6952, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8655, 0.8655, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.9532, 0.9532, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8607, 0.8607, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5902, 0.5902, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5562, 0.5562, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8947, 0.8947, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6320, 0.6320, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2846, 0.2846, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.6409, 0.6409, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3248, 0.3248, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8513, 0.8513, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0619, 0.0619, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0267, 0.0267, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3446, 0.3446, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
#        device='cuda:0', dtype=torch.float64), 'output_decoder_index': 
# tensor([[4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0],
#         [4, 5, 0, 0, 0, 0, 0, 0]], device='cuda:0')}, 'target_values': 
# {'drifting_gratings_temporal_frequency': tensor([4, 1, 1, 4, 0, 3, 3, 2, 3, 4, 
# 0, 3, 0, 2, 1, 0, 0, 4, 1, 2, 1, 0, 1, 2,
#         4, 3, 3, 2, 0, 1, 3, 2, 1, 2, 1, 3, 2, 4, 1, 0, 0, 1, 0, 1, 4, 0, 3, 2,
#         4, 4, 4, 3, 1, 2, 2, 4, 0, 2, 0, 3, 2, 4, 2, 4], device='cuda:0'), 
# 'drifting_gratings_orientation': tensor([3, 4, 2, 6, 0, 4, 0, 7, 6, 0, 1, 6, 3,
# 6, 3, 2, 0, 1, 1, 7, 4, 5, 4, 4,
#         1, 7, 0, 0, 6, 1, 3, 3, 4, 4, 1, 5, 2, 3, 0, 4, 1, 1, 2, 6, 5, 6, 7, 1,
#         7, 3, 0, 7, 2, 5, 0, 4, 6, 6, 5, 7, 1, 6, 6, 7], device='cuda:0')}, 
# 'target_weights': {'drifting_gratings_temporal_frequency': tensor([1., 1., 1., 
# 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0'), 
# 'drifting_gratings_orientation': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 
# 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')}, 
# 'session_id': ['allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563', 
# 'allen_visual_coding_ophys_2016/710504563'], 'absolute_start': 
# tensor([2018.1108,   53.8092, 1777.1660,  182.9462, 1666.2232,   60.3526,
#         1945.9096, 1750.3845,  282.2200,  294.8395,   84.0392, 1646.0059,
#          387.0686,  219.1825,  321.9254, 1846.4333,  240.5195,   65.8783,
#          261.3562,  213.0021,  411.7332, 1771.8529,  483.0987, 1996.4327,
#          393.2542, 1885.8654,  462.9597, 1699.4065, 1732.9171, 1840.2535,
#         1927.6163,  215.9537,  129.4868,  333.3199,  369.1971, 2035.4913,
#          564.9733,  198.7999,  287.9910, 1660.5602,  507.4179,  423.9519,
#         2021.0645,  444.5276,  300.6687,   96.4905, 2023.8578,  594.4753,
#         1636.4932,  489.3849,  104.9054, 1981.3324,  194.9867,  381.4112,
#          504.5506, 1843.2892, 1891.5916,  567.8543,  498.4486, 1999.9798,
#         1801.2938, 1952.2030, 1937.2278,  528.7649], device='cuda:0',
#        dtype=torch.float64), 'eval_mask': 
# {'drifting_gratings_temporal_frequency': tensor([True, True, True, True, True, 
# True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True], device='cuda:0'), 
# 'drifting_gratings_orientation': tensor([True, True, True, True, True, True, 
# True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True], device='cuda:0')}}



        # forward pass
        output_values = self.model(**batch["model_inputs"], unpack_output=False)

        # compute loss
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        taskwise_loss = {}
        for readout_id in output_values.keys():
            output = output_values[readout_id]
            target = target_values[readout_id]

            spec = self.model.readout.readout_specs[readout_id]

            weights = 1.0
            if readout_id in target_weights and target_weights[readout_id] is not None:
                weights = target_weights[readout_id]

            taskwise_loss[readout_id] = spec.loss_fn(output, target, weights)

            # count the number of sequences in the batch that have the current task
            num_sequences_with_current_task = torch.any(
                batch["model_inputs"]["output_decoder_index"]
                == MODALITY_REGISTRY[readout_id].id,
                dim=1,
            ).sum()
            loss = loss + taskwise_loss[readout_id] * num_sequences_with_current_task

        batch_size = batch["model_inputs"]["input_unit_index"].shape[0]
        # TODO change batch_size when POYOPlusEfficient is used
        loss = loss / batch_size

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        unit_index = batch["model_inputs"]["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss #! pytorch lightning handles backprop.

    def validation_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"], unpack_output=True)

        # prepare data for evaluator
        # (goes to MultiTaskDecodingStitchEvaluator.on_validation_batch_end)
        data_for_eval = DataForMultiTaskDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            decoder_indices=batch["model_inputs"]["output_decoder_index"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

        return data_for_eval

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        #import pdb; pdb.set_trace()
        self.cfg = cfg #* cfg contains multitask_readout
        self.log = logging.getLogger(__name__)
#(Pdb)  cfg
#{'data_root': '../poyo/data/processed/', 'log_dir': './logs', 
# 'train_transforms': [{'_target_': 'torch_brain.transforms.UnitDropout', 'field': 'calcium_traces.df_over_f', 'min_units': 10, 'mode_units': 50, 'max_units': 300, 'tail_right': 100}], 
# 'eval_transforms': [], 'epochs': 100, 'batch_size': 64, 'eval_epochs': 1, 'eval_batch_size': None, 'seed': 42, 'sanity_check_validation': False, 
# 'optim': {'base_lr': 3.125e-05, 'weight_decay': 0.0001, 'lr_decay_start': 0.5}, 'wandb': {'enable': False, 'entity': None, 'project': 'poyo', 'run_name': 'capoyo_single_session', 'log_model': False}, 
# 'precision': 32, 'nodes': 1, 'gpus': 1, 'num_workers': 1, 'ckpt_path': None, 
# 'model': {'_target_': 'torch_brain.models.CaPOYO', 'sequence_length': 1.0, 'latent_step': 0.125, 'num_latents_per_step': 16, 'dim': 64, 'dim_head': 64, 'depth': 6, 'cross_heads': 2, 'self_heads': 8, 'ffn_dropout': 0.2, 'lin_dropout': 0.4, 'atn_dropout': 0.2}, 
# 'dataset': [{'selection': [{'brainset': 'allen_visual_coding_ophys_2016', 'sessions': ['710504563']}], 'config': {'sampling_intervals_modifier': 'sampling_intervals = sampling_intervals & data.drifting_gratings.coalesce(0.3)\n', 
# 'multitask_readout': [{'readout_id': 'drifting_gratings_orientation', 'timestamp_key': 'drifting_gratings.timestamps', 'value_key': 'drifting_gratings.orientation_id', 'metrics': [{'metric': {'_target_': 'torchmetrics.Accuracy', 'task': 'multiclass', 'num_classes': 8}}]}, {'readout_id': 'drifting_gratings_temporal_frequency', 'timestamp_key': 'drifting_gratings.timestamps', 'value_key': 'drifting_gratings.temporal_frequency_id', 'metrics': [{'metric': {'_target_': 'torchmetrics.Accuracy', 'task': 'multiclass', 'num_classes': 5}}]}]}}]}



    def setup_dataset_and_link_model(self, model: POYOPlus):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """
        self.sequence_length = model.sequence_length

        # prepare transforms
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)

        # compose transforms, tokenizer is always the last transform
        train_transform = Compose([*train_transforms, model.tokenize]) #! Compose : sequentially apply transform. => tokenize. tokenize is dependent on model
        
        """
# First, get a recording_id from the dataset
recording_id = list(self.train_dataset.recording_dict.keys())[0]

# Get the sampling intervals to find valid start/end times
intervals = self.train_dataset.get_sampling_intervals()[recording_id]

# Create the DatasetIndex (import it first)
from torch_brain.data.dataset import DatasetIndex

# Create index with start from the first interval, using a small window (e.g., 1.0 second)
idx = DatasetIndex(recording_id=recording_id, start=intervals.start[0], end=intervals.start[0] + 1.0)

# Now access the item
item = self.train_dataset[idx]
print(item)

(Pdb) item.keys()
dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
(Pdb) item['model_inputs'].keys()
dict_keys(['input_unit_index', 'input_timestamps', 'input_values', 'input_mask', 'latent_index', 'latent_timestamps', 'output_session_index', 'output_timestamps', 'output_decoder_index'])

output_session_index': Padded8Object(obj=array([1, 1])), 'output_timestamps': Padded8Object(obj=tensor([0.9807, 0.9807], dtype=torch.float64)), 'output_decoder_index': Padded8Object(obj=tensor([4, 5]))}, 'target_values': ChainObject(obj={'drifting_gratings_orientation': array([5]), 'drifting_gratings_temporal_frequency': array([2])}, allow_missing_keys=True), 'target_weights': ChainObject(obj={'drifting_gratings_orientation': array([1.], dtype=float32), 'drifting_gratings_temporal_frequency': array([1.], dtype=float32)}, allow_missing_keys=True), 'session_id': 'allen_visual_coding_ophys_2016/710504563', 'absolute_start': 32.73288, 'eval_mask': ChainObject(obj={'drifting_gratings_orientation': array([ True]), 'drifting_gratings_temporal_frequency': array([ True])}, allow_missing_keys=True)}
 (Pdb) type(item['model_inputs']['latent_timestamps'])
<class 'numpy.ndarray'>
(Pdb) type(item['model_inputs']['input_timestamps'])
<class 'torch_brain.data.collate.Padded8Object'>
(Pdb) type(item['model_inputs']['output_timestamps'])
<class 'torch_brain.data.collate.Padded8Object'>
        """

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=train_transform,
        )
        self.train_dataset.disable_data_leakage_check()
#(Pdb) print(self.train_dataset)
#Dataset(root=../poyo/data/processed/, config=[{'selection': [{'brainset': 'allen_visual_coding_ophys_2016', 'sessions': ['710504563']}], 'config': {'sampling_intervals_modifier': 'sampling_intervals = sampling_intervals & data.drifting_gratings.coalesce(0.3)\n', 'multitask_readout': [{'readout_id': 'drifting_gratings_orientation', 'timestamp_key': 'drifting_gratings.timestamps', 'value_key': 'drifting_gratings.orientation_id', 'metrics': [{'metric': {'_target_': 'torchmetrics.Accuracy', 'task': 'multiclass', 'num_classes': 8}}]}, {'readout_id': 'drifting_gratings_temporal_frequency', 'timestamp_key': 'drifting_gratings.timestamps', 'value_key': 'drifting_gratings.temporal_frequency_id', 'metrics': [{'metric': {'_target_': 'torchmetrics.Accuracy', 'task': 'multiclass', 'num_classes': 5}}]}]}}], split=train)

        #sample train_dataset item
        #dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
        #model_inputs keys: dict_keys(['input_unit_index', 'input_timestamps', 'input_values', 'input_mask', 'latent_index', 'latent_timestamps', 'output_session_index', 'output_timestamps', 'output_decoder_index'])
        self._init_model_vocab(model) #!* For myself ) shd change model_input keys to match. #*! DataModule 에서 얼마나 많이 바꿔야되는지를 확인할 것. 많이 바꿔야한다면 custom dmfh rngusgksms rp skdma. 
#(Pdb) model
# CaPOYO(
#   (input_value_map): Linear(in_features=1, out_features=32, bias=True)
#   (unit_emb): InfiniteVocabEmbedding(embedding_dim=32, num_embeddings=0)
#   (session_emb): InfiniteVocabEmbedding(embedding_dim=64, num_embeddings=0)
#   (token_type_emb): Embedding(4, 64)
#   (task_emb): Embedding(20, 64)
#   (latent_emb): Embedding(16, 64)
#   (rotary_emb): RotaryTimeEmbedding()
#   (dropout): Dropout(p=0.4, inplace=False)
#   (enc_atn): RotaryCrossAttention(
#     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (norm_context): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (to_q): Linear(in_features=64, out_features=128, bias=False)
#     (to_kv): Linear(in_features=64, out_features=256, bias=False)
#     (to_out): Linear(in_features=128, out_features=64, bias=True)
#   )
#   (enc_ffn): Sequential(
#     (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (1): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=64, out_features=512, bias=True)
#         (1): GEGLU()
#         (2): Dropout(p=0.2, inplace=False)
#         (3): Linear(in_features=256, out_features=64, bias=True)
#       )
#     )
#   )
#   (proc_layers): ModuleList(
#     (0-5): 6 x Sequential(
#       (0): RotarySelfAttention(
#         (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#         (to_qkv): Linear(in_features=64, out_features=1536, bias=False)
#         (to_out): Linear(in_features=512, out_features=64, bias=True)
#       )
#       (1): Sequential(
#         (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#         (1): FeedForward(
#           (net): Sequential(
#             (0): Linear(in_features=64, out_features=512, bias=True)
#             (1): GEGLU()
#             (2): Dropout(p=0.2, inplace=False)
#             (3): Linear(in_features=256, out_features=64, bias=True)
#           )
#         )
#       )
#     )
#   )
#   (dec_atn): RotaryCrossAttention(
#     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (norm_context): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (to_q): Linear(in_features=64, out_features=128, bias=False)
#     (to_kv): Linear(in_features=64, out_features=256, bias=False)
#     (to_out): Linear(in_features=128, out_features=64, bias=True)
#   )
#   (dec_ffn): Sequential(
#     (0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#     (1): FeedForward(
#       (net): Sequential(
#         (0): Linear(in_features=64, out_features=512, bias=True)
#         (1): GEGLU()
#         (2): Dropout(p=0.2, inplace=False)
#         (3): Linear(in_features=256, out_features=64, bias=True)
#       )
#     )
#   )
#   (readout): MultitaskReadout(
#     (projections): ModuleDict(
#       (cursor_velocity_2d): Linear(in_features=64, out_features=2, bias=True)
#       (cursor_position_2d): Linear(in_features=64, out_features=2, bias=True)
#       (arm_velocity_2d): Linear(in_features=64, out_features=2, bias=True)
#       (drifting_gratings_orientation): Linear(in_features=64, out_features=8, bias=True)
#       (drifting_gratings_temporal_frequency): Linear(in_features=64, out_features=5, bias=True)
#       (natural_movie_one_frame): Linear(in_features=64, out_features=900, bias=True)
#       (natural_movie_two_frame): Linear(in_features=64, out_features=900, bias=True)
#       (natural_movie_three_frame): Linear(in_features=64, out_features=3600, bias=True)
#       (locally_sparse_noise_frame): Linear(in_features=64, out_features=8000, bias=True)
#       (static_gratings_orientation): Linear(in_features=64, out_features=6, bias=True)
#       (static_gratings_spatial_frequency): Linear(in_features=64, out_features=5, bias=True)
#       (static_gratings_phase): Linear(in_features=64, out_features=5, bias=True)
#       (natural_scenes): Linear(in_features=64, out_features=119, bias=True)
#       (gabor_orientation): Linear(in_features=64, out_features=4, bias=True)
#       (gabor_pos_2d): Linear(in_features=64, out_features=2, bias=True)
#       (running_speed): Linear(in_features=64, out_features=1, bias=True)
#       (gaze_pos_2d): Linear(in_features=64, out_features=2, bias=True)
#       (pupil_location): Linear(in_features=64, out_features=2, bias=True)
#       (pupil_size_2d): Linear(in_features=64, out_features=2, bias=True)
#       (eeg_reconstruction): Linear(in_features=64, out_features=1, bias=True)
#     )
#   )
# )
        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        # validation and test datasets require a tokenizer that is in eval mode
        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.val_dataset.disable_data_leakage_check()

        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.test_dataset.disable_data_leakage_check()

    def _init_model_vocab(self, model: POYOPlus):
        # TODO: Add code for finetuning situation (when model already has a vocab)
        model.unit_emb.initialize_vocab(self.get_unit_ids())
        model.session_emb.initialize_vocab(self.get_session_ids())

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def get_multitask_readout_registry(self):
        config_dict = self.train_dataset.get_recording_config_dict()

        custum_readout_registry = {}
        for recording_id in config_dict.keys():
            config = config_dict[recording_id]
            multitask_readout = config["multitask_readout"]

            for readout_config in multitask_readout:
                readout_id = readout_config["readout_id"]
                if readout_id not in MODALITY_REGISTRY:
                    raise ValueError(
                        f"Readout {readout_id} not found in modality registry, please register it "
                        "using torch_brain.register_modality()"
                    )
                custum_readout_registry[readout_id] = MODALITY_REGISTRY[readout_id]
        return custum_readout_registry

    def get_metrics(self):
        dataset_config_dict = self.get_recording_config_dict()
        metrics = defaultdict(lambda: defaultdict(dict))
        # setup the metrics
        for recording_id, recording_config in dataset_config_dict.items():
            for readout_config in recording_config["multitask_readout"]:
                readout_id = readout_config["readout_id"]
                for metric_config in readout_config["metrics"]:
                    metric = hydra.utils.instantiate(metric_config["metric"])
                    metrics[recording_id][readout_id][str(metric)] = metric
        return metrics

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        return train_loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        self.val_sequence_index = val_sampler.sequence_index

        return val_loader

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml") #DIVER model builds via cfg.py while POYO_plus via config/*yaml
def main(cfg: DictConfig):
    logger.info("POYO+!")
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # make model and datamodule
    # TODO: resolve the readout_id from dataset, only build readouts needed
    #import pdb; pdb.set_trace()
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY) # model 생성, yaml config 에서 _target_ 키를 보고 해당 클래스 자동 instance 화
    data_module = DataModule(cfg)
    data_module.setup_dataset_and_link_model(model)

    # Lightning train wrapper
    #import pdb; pdb.set_trace()
    wrapper = TrainWrapper(cfg=cfg, model=model)

    evaluator = MultiTaskDecodingStitchEvaluator(metrics=data_module.get_metrics()) #! TODO : MultiTaskDecodingStitchEvaluator 알아보기!

    callbacks = [ #! PyTorch Lightning Callbacks입니다. 학습 중 특정 시점에 자동으로 실행되는 함수들
        evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint( #Save best model checkpoint based on average_val_metric, also save last.ckpt
            save_last=True,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor( #log lr -> wandb / tensorboard
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
        tbrain_callbacks.MemInfo(), #GPU memory usage logging -> debug OOM
        tbrain_callbacks.EpochTimeLogger(), #log epoch time 
        tbrain_callbacks.ModelWeightStatsLogger(), #log model weight and gradient stats -> debug gradient vanishing, exploding
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,  # Ensure no limit on validation batches
        num_sanity_val_steps=-1 if cfg.sanity_check_validation else 0,
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
