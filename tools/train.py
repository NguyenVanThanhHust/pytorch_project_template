# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
@modified: thanh
@contact: nguyenvanthanhhust@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
from srcs.config import cfg
from srcs.data import make_data_loader
from srcs.engine.trainer import do_train
from srcs.modeling import build_model
from srcs.solver import make_optimizer

from srcs.utils.logger import setup_logger


def train(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter()

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        writer
    )
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Template Mini Imagenet Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs("outputs", exist_ok=True)
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("main_process", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()
