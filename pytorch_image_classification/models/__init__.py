import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config
from ..transforms import _get_dataset_stats
import numpy as np
from typing import *


def create_model(config: yacs.config.CfgNode) -> nn.Module:
    device = torch.device(config.device)
    module = importlib.import_module(
        'pytorch_image_classification.models'
        f'.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    if config.model.normalize_layer:
        mean, std = _get_dataset_stats(config)
        mean = torch.from_numpy(mean).float().to(device)
        std = torch.from_numpy(std).float().to(device)
        normalize_layer=NormalizeLayer(mean,std)
        model=torch.nn.Sequential(normalize_layer, model)
    model.to(device)
    return model


def apply_data_parallel_wrapper(config: yacs.config.CfgNode,
                                model: nn.Module) -> nn.Module:
    local_rank = config.train.dist.local_rank
    if dist.is_available() and dist.is_initialized():
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    else:
        model.to(config.device)
    return model

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: torch.Tensor, sds: torch.Tensor):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = means
        self.sds = sds

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds