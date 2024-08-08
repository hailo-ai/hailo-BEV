#!/usr/bin/env python3
# pylint: disable=C0301
# pylint: disable=E0611
# pylint: disable=C0411
# pylint: disable=W0621

import torch
import numpy as np
import sys
import os
import argparse

from mmcv.utils import import_modules_from_strings
from mmcv import Config

from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset

def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument("-p", "--path", default="PETR/", help="path to the PETR folder.")
    parser.add_argument("-c", "--cfg", default="resources/config_file.py", help="path to the config file.")
    parsed_args = parser.parse_args()
    return parsed_args

def load_cfg(cfg_path, petr_path, samples_per_gpu=1):
    """
    Load a configuration file and perform necessary configurations.

    Args:
        cfg_path (str): Path to the configuration file.
        samples_per_gpu (int, optional): Number of samples per GPU. Default is 1.

    Returns:
        mmcv.Config: Modified configuration object loaded from `cfg_path`.
    """
    sys.path.insert(0, petr_path)
    cfg = Config.fromfile(cfg_path)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:

            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
            else:
                _module_dir = os.path.dirname(cfg_path)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)


    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    return cfg

def save_timestamp(img_metas, sample) -> None:
    """
    Calculate mean timestamps from image metadata and save as a PyTorch tensor.
    """
    time_stamps = []
    for img_meta in img_metas:
        time_stamps.append(np.asarray(img_meta['timestamp']))
    time_stamp = torch.tensor(time_stamps)
    time_stamp = time_stamp.view(1, -1, 6)
    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
    torch.save(mean_time_stamp.cpu(), f'resources/input/data_{sample}.pt')

def save_img2lidars(img_metas, sample) -> None:
    """
    Calculate mean img2lidars from image metadata and save as a PyTorch tensor.
    """
    img2lidars = []
    for img_meta in img_metas:
        img2lidar = []
        for i in range(len(img_meta['lidar2img'])):
            img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
        img2lidars.append(np.asarray(img2lidar))
    img2lidars = np.asarray(img2lidars)
    img2lidars = torch.tensor(img2lidars)
    img2lidars = img2lidars = img2lidars.view(1, 12, 1, 1, 1, 4, 4).repeat(1, 1, 25, 10, 64, 1, 1)
    torch.save(img2lidars.cpu(), f'resources/input/img2lidars_{sample}.pt')

if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.cfg, args.path, samples_per_gpu=1)
    #create data after pre process
    print("Building dataset")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        shuffle=False)
    dataset = data_loader.dataset
    print("Dataset Built")

    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data[0]
        sample = img_metas[0]['sample_idx']
        torch.save(data['img'][0].data[0], f'resources/input/data_{sample}.pt')
        save_timestamp(img_metas, sample)
        save_img2lidars(img_metas, sample)
