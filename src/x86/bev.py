#!/usr/bin/env python3
# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=W0621
# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0413

import numpy as np
import threading
import time
import argparse
import queue
import json
import sys
import os
from hailo_platform import (Device, VDevice, HailoSchedulingAlgorithm)

sys.path.append('src/common')
import fps_calc
import multiprocessing
import pre_post_process
import core
import demo_manager
import async_api
import visualization

MAX_QUEUE_SIZE = 3

def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument('--run-slow', action='store_true', help='Run the demo at 5 FPS for better visualization of 3D boxes.')
    parser.add_argument('--raw-data', action='store_true', help=' Run the demo from raw data for lower cpu usage.')
    parser.add_argument("-i", "--input", default="resources/input/", help="Path to the input folder, Use this flag only if you have modified the default input folder location.")
    parser.add_argument("-m", "--models", default="resources/models/", help="Path to the models folder, Use this flag only if you have modified the default models folder location.")
    parser.add_argument("-d", "--data", default="resources/data/", help="Path to the data folder, where the nuScenes dataset is.")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = parse_args()
    with open(f'{args.input}/nusc_tiny_dataset.json', 'r', encoding='utf-8') as file:
        nusc = json.load(file)

    backbone_hef_path = f'{args.models}petrv2_repvggB0_backbone_pp_800x320.hef'
    transformer_hef_path = f'{args.models}/petrv2_repvggB0_transformer_pp_800x320.hef'
    post_proc_onnx_path = f'{args.models}/petrv2_postprocess.onnx'
    matmul_path = f'{args.models}/matmul.npy'

    fps_calculator = fps_calc.FPSCalc(2)

    queues = []
    bb_tranformer_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_meta_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    bb_tranformer_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    transformer_pp_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    pp_3dnms_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    nms_send_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

    queues.append(bb_tranformer_meta_queue)
    queues.append(transformer_pp_meta_queue)
    queues.append(bb_tranformer_queue)
    queues.append(transformer_pp_queue)
    queues.append(pp_3dnms_queue)
    queues.append(nms_send_queue)

    with open(f'{args.input}tokens.json', 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    scenes = scenes['scenes']

    tokens = []
    for scene in scenes:
        tokens.append(scene['tokens'])

    manager = multiprocessing.Manager()
    demo_mng = demo_manager.DemoManager(manager)
    devices = Device.scan()
    params = async_api.create_vdevice_params()
    threads = []
    processes = []
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    with VDevice(params) as target:
        if args.raw_data:
            threads.append(threading.Thread(target=core.backbone_raw_data, args=(target, args.input, backbone_hef_path, bb_tranformer_queue,
                                                    bb_tranformer_meta_queue, demo_mng, scenes, args.run_slow)))
        else:
            threads.append(threading.Thread(target=core.backbone_from_jpg, args=(target, args.data, backbone_hef_path, bb_tranformer_queue,
                                        bb_tranformer_meta_queue, demo_mng, scenes, args.run_slow, nusc)))

        threads.append(threading.Thread(target=core.transformer, args=(target, transformer_hef_path, matmul_path, bb_tranformer_queue, bb_tranformer_meta_queue, transformer_pp_queue, transformer_pp_meta_queue, demo_mng)))

        processes.append(multiprocessing.Process(target=pre_post_process.post_proc,
                                                args=(transformer_pp_queue, transformer_pp_meta_queue,
                                                pp_3dnms_queue, post_proc_onnx_path, demo_mng)))
        processes.append(multiprocessing.Process(target=pre_post_process.d3nms_proc,
                                                args=(pp_3dnms_queue, nms_send_queue, nusc, demo_mng)))
        processes.append(multiprocessing.Process(target=visualization.viz_proc,
                                            args=(args.input, args.data, nms_send_queue,
                                            fps_calculator, nusc)))

        try:
            for process in processes:
                process.start()

            for thread in threads:
                thread.daemon = True
                thread.start()

            while not demo_mng.get_terminate():
                time.sleep(1)


        except KeyboardInterrupt:
            demo_mng.set_terminate()

        finally:
            for thread in threads:
                thread.join()

            for process in processes:
                process.join()

            for queue_to_close in queues:
                queue_to_close.close()

            target.release()
