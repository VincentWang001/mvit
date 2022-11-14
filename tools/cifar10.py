# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""

import argparse
import sys

import mvit.utils.checkpoint as cu
from engine import test, train
from mvit.config.defaults import assert_and_infer_cfg, get_cfg
from mvit.utils.misc import launch_job

import pprint

import mvit.models.losses as losses
import mvit.models.optimizer as optim
import mvit.utils.checkpoint as cu
import mvit.utils.distributed as du
import mvit.utils.logging as logging
import mvit.utils.metrics as metrics
import mvit.utils.misc as misc
import numpy as np
import torch
from mvit.datasets import loader
from mvit.datasets.mixup import MixUp
from mvit.models import build_model
from mvit.utils.meters import EpochTimer, TrainMeter, ValMeter

logger = logging.get_logger(__name__)

from main import parse_args, load_config

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # print("START OF CFG")
    # print(cfg)
    # print("END OF CFG")

    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)
    #model state loaded
    print("MODEL LOADED SUCCESSFULLY") 

if __name__ == "__main__":
    main()
