#!/usr/bin/env python3
# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=W0621
# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=C0413
import time
import argparse
import os
import multiprocessing
import sys
import json
sys.path.append('src/common')

import fps_calc
import visualization
import client

MAX_QUEUE_SIZE = 5

def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument('--set-port', default=5555, type=int, help="Change the port from 5555 to another one.")
    parser.add_argument('--set-ip', default='127.0.0.1', help="Set platform's ip.")
    parser.add_argument("-i", "--input", default="resources/input/", help="Path to the input folder, Path to the input folder, Use this flag only if you have modified the default input folder location.")
    parser.add_argument("-d", "--data", default="resources/data/", help="Path to the data folder, where the nuScenes dataset is.")
    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == "__main__":
    args = parse_args()
    d3nms_out_vis_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    with open(f'{args.input}/nusc_tiny_dataset.json', 'r', encoding='utf-8') as file:
        nusc = json.load(file)
    fps_calculator = fps_calc.FPSCalc(2)


    visualize_process = multiprocessing.Process(target=visualization.viz_proc,
                                            args=(args.input, args.data, d3nms_out_vis_in_queue,
                                            fps_calculator, nusc))

    get_process = multiprocessing.Process(target=client.start_client,
                                            args=(d3nms_out_vis_in_queue, args.set_port, args.set_ip))

    try:
        get_process.start()
        visualize_process.start()
        time.sleep(1000000000)

    except KeyboardInterrupt:
        get_process.terminate()
        visualize_process.terminate()
        d3nms_out_vis_in_queue.close()
        os._exit(0)
