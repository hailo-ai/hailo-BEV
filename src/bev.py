#!/usr/bin/env python3
# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=W0621
# pylint: disable=C0103

import numpy as np
import torch
import cv2
import time
import argparse
from typing import Union, List, Tuple, Any
import os

from nuscenes.nuscenes import NuScenes

from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, HailoSchedulingAlgorithm, FormatType)

import multiprocessing
import visualization
import middle_post_process
import fps_calc

MAX_QUEUE_SIZE = 5

def create_vdevice_params():
    """
    Create parameters for a virtual device.
    """
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = True
    return params

def bb_send(network_group, queue, infinite_loop) -> None:
    """
    Send input data to a network group of virtual streams.

    This function initializes input streams using `InputVStreamParams.make_from_network_group()`
    with `FormatType.FLOAT32` format, and sends data in a loop to each stream in the network group.

    Parameters:
        network_group (object): The network group to send input data to.
    """
    bb_input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)
    with InputVStreams(network_group, bb_input_vstreams_params) as vstreams:
        while True:
            for _ in range(iterations_num):
                for vstream in vstreams:
                    in_data = queue.get()
                    input_data = np.transpose(in_data[0], (0, 2, 3, 1)) # N C H W -> N H W C
                    input_data = np.ascontiguousarray(input_data)
                    vstream.send(input_data)

            if not infinite_loop:
                break

def bb_recv(network_group, queue, infinite_loop) -> None:
    """
    Receive backbone data from video streams and put processed results into a queue.

    Args:
    - network_group (object): Object representing the network group for video streams.
    - queue (Queue object): Queue where processed results are put.

    Returns:
    - None

    Description:
    This function continuously receives backbone data from multiple video streams (vstreams)
    within a network group. Each vstream sends 12 data segments sequentially, which are then
    stacked and reshaped before being placed into the provided queue for further processing.
    """
    bb_output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group,
                                                                    format_type=FormatType.FLOAT32)
    with OutputVStreams(network_group, bb_output_vstreams_params) as vstreams:
        while True:
            for _ in range(iterations_num):
                output_data = {}
                for vstream in vstreams:
                    output_data[vstream.name] = []
                for _ in range(12):
                    for vstream in vstreams:
                        output_data[vstream.name].append(vstream.recv())

                result1 = np.transpose(np.array(output_data['petrv2_b0_backbone_x32_BN_q_304_dec_3_UN_800x320/conv31']), (0, 3, 1, 2)) #N H W C ->  N C H W
                result2 = np.transpose(np.array(output_data['petrv2_b0_backbone_x32_BN_q_304_dec_3_UN_800x320/conv29']), (0, 3, 1, 2)) #N H W C ->  N C H W

                queue.put((result1, result2)) # backbone output

            if not infinite_loop:
                break

def transformer_send(network_group, queue, infinite_loop) -> None:
    """
    Send data from a queue to transformer hef.

    Args:
    - network_group (object): Object representing the network group for video streams.
    - queue (Queue object): Queue containing data to be sent to video streams.

    Returns:
    - None

    Description:
    This function continuously retrieves data from the provided queue and sends it to
    corresponding input video streams (vstreams) within a network group. The data from
    the queue is transposed and expanded before being sent to each vstream.
    """
    t_input_vstreams_params=  InputVStreamParams.make_from_network_group(network_group,
                                                                    format_type=FormatType.FLOAT32)
    with InputVStreams(network_group, t_input_vstreams_params) as vstreams:
        while True:
            for _ in range(iterations_num):
                in_data = queue.get()
                for j, vstream in enumerate(vstreams):
                    vstream.send(np.expand_dims(np.transpose(in_data[j], (1, 0, 2)), axis=0))

            if not infinite_loop:
                break

def transformer_recv(network_group,queue, infinite_loop) -> None:
    """
    Receive data from the transformer hef and put processed results into a queue.

    Args:
    - network_group (object): Object representing the network group for video streams.
    - queue (Queue object): Queue where processed results are put.

    Returns:
    - None

    Description:
    This function continuously receives data from multiple video streams (vstreams)
    within a network group. The received data is processed and stacked into a specific
    order before being placed into the provided queue for further processing or 
    consumption by other parts of the application.
    """
    t_output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group,
                                                                    format_type=FormatType.FLOAT32)
    with OutputVStreams(network_group, t_output_vstreams_params) as vstreams:
        while True:
            for _ in range(iterations_num):
                output_data = {}
                for vstream in vstreams:
                    output_data[vstream.name] = vstream.recv()

                result=np.stack((output_data['petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_const0/mul_and_add5'],
                                    output_data['petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_const0/mul_and_add10'],
                                    output_data['petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_const0/mul_and_add15']), axis=0)
                queue.put(result)

            if not infinite_loop:
                break

def configure_and_get_network_group(hef, target):
    """
    Configure and retrieve a network group from a target device.
    """
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    return network_group

def check_fps_range(value) -> Union[str, int]:
    """
    Validate and return an integer FPS value within the range of 1 to 10 (inclusive).

    Args:
        value (str or int): The FPS value to validate.

    Returns:
        int: Validated FPS value within the range of 1 to 10.
    """
    ivalue = int(value)
    if ivalue < 1 or ivalue > 10:
        raise argparse.ArgumentTypeError(f"FPS must be between 1 and 10 (inclusive), got {value}")
    return ivalue

def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument("-f", "--fps", default=-1, type=check_fps_range, required=False, help="wanted FPS (1 - 10).")
    parser.add_argument('--infinite-loop', action='store_true', help='run the demo in infinite loop.')
    parser.add_argument("-i", "--input", default="resources/input/", help="path to the input folder.")
    parser.add_argument("-m", "--models", default="resources/models/", help="path to the models folder.")
    parser.add_argument("-d", "--data", default="resources/data/", help="path to the data folder, where the nuScenes dataset is.")
    parser.add_argument("-n", "--number-of-scenes", default="2", type=int, help="number of scenes to run")

    parsed_args = parser.parse_args()
    return parsed_args

def create_token_and_image(iterations_num):
    """
    Create lists of tokens and images, each initialized with None, based on the given number of iterations.
    """
    token = [None]*iterations_num
    img = [None]*iterations_num
    return token, img

def create_data(iterations_num) -> Tuple[List[None], List[None], List[None], List[Any], List[Any], List[None], List[None], List[None], List[None]]:
    """
    Create and initialize lists for various types of data based on the given number of iterations.
    """
    timestamp = [None]*iterations_num
    img2lidars = [None]*iterations_num
    tensor_data = [None]*iterations_num

    cams_token = []
    cams_images = []

    for _ in range(7):
        token, img = create_token_and_image(iterations_num)
        cams_token.append(token)
        cams_images.append(img)


    pose_record = [None]*iterations_num
    cam_intrinsic = [None]*iterations_num

    sd_record = [None]*iterations_num
    cs_record = [None]*iterations_num
    return timestamp, img2lidars, tensor_data, cams_token, cams_images, pose_record, cam_intrinsic, sd_record, cs_record

def load_image(sample, cam, cam_token, cam_images) -> None:
    """
    Load an image from the given sample data for a specific camera.
    """
    sd_rec = nusc.get('sample_data', sample['data'][cam])
    cam_token[index] = sd_rec['token']
    if cam != 'LIDAR_TOP':
        path = sd_rec['filename']
        cam_images[index] = cv2.imread(f'{args.data}/{path}')
        cam_images[index] = cv2.cvtColor(cam_images[index], cv2.COLOR_BGR2RGB)

def load_images(sample, cams_token, cams_images) -> None:
    """
    Load and preprocess images for multiple cameras in a given sample.
    """
    cams = ['CAM_FRONT_LEFT','CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
    for cam_token, cam_images, cam in zip(cams_token, cams_images, cams):
        load_image(sample, cam, cam_token, cam_images)

def get_scene_tokens(i, nusc) -> List[str]:
    """
    Retrieve sample tokens associated with a specific scene from the NuScenes dataset.
    """
    scenes = nusc.scene
    scene = scenes[i]
    scene_token = scene['token']
    # Get sample tokens for the scene
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
    return sample_tokens

if __name__ == "__main__":
    args = parse_args()
    backbone_hef_path = f'{args.models}/petrv2_b0_backbone_x32_BN_q_304_dec_3_UN_800x320.hef'
    mid1_proc_onnx_path = f'{args.models}/petrv2_middle_process_1.onnx'
    mid2_proc_onnx_path = f'{args.models}/petrv2_middle_process_2.onnx'
    
    transformer_hef_path = f'{args.models}/petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_const0.hef'
    post_proc_onnx_path = f'{args.models}/petrv2_post_process.onnx'

    bb_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    bb_out_mid_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    mid_out_trans_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    trans_out_pp_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    pp_out_3dnms_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    d3nms_out_vis_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)
    token_queue = multiprocessing.Queue(maxsize=7 * MAX_QUEUE_SIZE)
    vis_out_top_in_queue = multiprocessing.Queue(maxsize=MAX_QUEUE_SIZE)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.data, verbose=False)

    tokens = []
    for i in range(args.number_of_scenes):
        tokens += get_scene_tokens(i, nusc)

    iterations_num = len(tokens)
    infinite_loop = args.infinite_loop
    wanted_fps = args.fps
    fps_calculator = fps_calc.FPSCalc(10)

    backbone_hef = HEF(backbone_hef_path)
    transformer_hef = HEF(transformer_hef_path)

    timestamp, img2lidars, tensor_data, cams_token, cams_images, pose_record, cam_intrinsic, sd_record, cs_record = create_data(iterations_num)

    classes = torch.load(f'{args.input}classes.pt')
    matmul = torch.load(f'{args.input}matmul.pt')


    devices = Device.scan()
    params = create_vdevice_params()

    with VDevice(params) as target:
        bb_network_group = configure_and_get_network_group(backbone_hef, target)
        bb_network_group_params = bb_network_group.create_params()
        T_network_group = configure_and_get_network_group(transformer_hef, target)
        T_network_group_params = T_network_group.create_params()
        bb_send_process = multiprocessing.Process(target=bb_send, args=(bb_network_group, bb_in_queue, infinite_loop))
        bb_recv_process = multiprocessing.Process(target=bb_recv, args=(bb_network_group,
                                                bb_out_mid_in_queue, infinite_loop))
        mid_process = multiprocessing.Process(target=middle_post_process.middle_proc,
                                                args=(bb_out_mid_in_queue,
                                                mid_out_trans_in_queue,
                                                iterations_num, infinite_loop,
                                                mid1_proc_onnx_path, mid2_proc_onnx_path, img2lidars,
                                                matmul, classes))
        transformer_send_process = multiprocessing.Process(target=transformer_send, args=(T_network_group,
                                                mid_out_trans_in_queue, infinite_loop))
        transformer_recv_process = multiprocessing.Process(target=transformer_recv, args=(T_network_group,
                                                trans_out_pp_in_queue, infinite_loop))
        post_process = multiprocessing.Process(target=middle_post_process.post_proc,
                                                args=(trans_out_pp_in_queue,
                                                pp_out_3dnms_in_queue, iterations_num,
                                                infinite_loop, post_proc_onnx_path, timestamp))
        d3nms_process = multiprocessing.Process(target=middle_post_process.d3nms_proc,
                                                args=(pp_out_3dnms_in_queue,d3nms_out_vis_in_queue,
                                                token_queue, iterations_num, infinite_loop, nusc))
        visualize_process = multiprocessing.Process(target=visualization.viz_proc,
                                                args=(d3nms_out_vis_in_queue,
                                                iterations_num, fps_calculator,
                                                pose_record, infinite_loop,
                                                cams_images, cams_token, nusc))

        index = 0
        for token in tokens:
            timestamp[index] = torch.load(f'{args.input}timestamp_{token}.pt')
            img2lidars[index] = torch.load(f'{args.input}img2lidars_{token}.pt').numpy()
            tensor_data[index] = torch.load(f'{args.input}data_{token}.pt')

            sample = nusc.get('sample', token)
            load_images(sample, cams_token, cams_images)

            sd_record[index] = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            pose_record[index] = nusc.get('ego_pose', sd_record[index]['ego_pose_token'])
            cs_record[index] = nusc.get('calibrated_sensor', sd_record[index]['calibrated_sensor_token'])
            cam_intrinsic[index] = np.array(cs_record[index]['camera_intrinsic'])

            index = index + 1
        try:
            visualize_process.start()
            bb_send_process.start()
            bb_recv_process.start()
            transformer_send_process.start()
            transformer_recv_process.start()
            post_process.start()
            d3nms_process.start()
            mid_process.start()
            start_time = time.time()

            ind = 0
            while True:
                for i, token in enumerate(tokens):
                    bb_in_queue.put(tensor_data[i])
                    token_queue.put(token)
                    ind = ind + 1
                    if wanted_fps > 0:
                        time.sleep(1/wanted_fps)

                if not infinite_loop:
                    break

        except KeyboardInterrupt:
            bb_recv_process.terminate()
            bb_send_process.terminate()
            mid_process.terminate()
            transformer_send_process.terminate()
            transformer_recv_process.terminate()
            post_process.terminate()
            d3nms_process.terminate()
            visualize_process.terminate()
            os._exit(0)
        if not infinite_loop:
            bb_recv_process.join()
            bb_send_process.join()
            mid_process.join()
            transformer_send_process.join()
            transformer_recv_process.join()
            post_process.join()
            d3nms_process.join()
            visualize_process.join()
            