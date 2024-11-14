#!/usr/bin/env python3
import numpy as np
from nuscenes.nuscenes import NuScenes
import argparse
import json
import sys
import cv2
sys.path.append('src/common')
import pre_post_process
import visualization

cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="BEV demo")
    parser.add_argument("-d", "--data", default="resources/data/", help="Path to the data folder, where the nuScenes dataset is.")
    parser.add_argument("-i", "--input", default="resources/input/", help="Path to the input folder. Use this flag only if you dont want to use the default input folder location.")
    parser.add_argument("-r", "--raw-data", action="store_true", default="False", help="Use this flag only if you want to run the demo from raw date input" )
    parsed_args = parser.parse_args()
    return parsed_args

def get_scene_tokens(i, nusc):
    """
    Retrieve sample tokens associated with a specific scene from the NuScenes dataset.
    """
    scenes = nusc.scene
    scene = scenes[i]
    scene_token = scene['token']
    # Get sample tokens for the scene
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
    return sample_tokens, scene_token

if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.data , verbose=False)
    data = {}
    data['scenes'] = []
    if args.raw_data == True:
        print('Preparing data for running from raw data')
    else:
        print('Preparing data for running from jpg')

    nusc_tiny_dataset = {}
    for i in [1, 3, 6, 8]:
        print(f'Preparing scene number {i}')
        sample_tokens, scene_token = get_scene_tokens(i, nusc)
        tokens = [f'first_{sample_tokens[0]}'] + sample_tokens

        input = []
        for i, token in enumerate(tokens):
            nusc_tiny_dataset[token] = {}
            if token.startswith('first'):
                sample = nusc.get('sample', token.removeprefix("first_"))
            else:
                sample = nusc.get('sample', token)

            file_paths = []
            for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']:
                sd_record = nusc.get('sample_data',sample['data'][cam])
                nusc_tiny_dataset[token][cam] = []
                nusc_tiny_dataset[token][cam].append(nusc.get('ego_pose', sd_record['ego_pose_token']))
                nusc_tiny_dataset[token][cam].append(nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token']))
                if cam != 'LIDAR_TOP':
                    file_paths.append(sd_record['filename'])
                    nusc_tiny_dataset[token][cam].append({"filename" : sd_record['filename']})
                else:
                    cv2.imwrite(f'{args.input}/map_files/{token}_LIDAR_TOP.jpg', visualization.render_ego_centric_map(nusc, sample['token']))
                    nusc_tiny_dataset[token][cam].append({"filename" : f'map_files/{token}_LIDAR_TOP.jpg'})
            input.append(pre_post_process.preprocess(args.data, file_paths))
        if args.raw_data == True:
            np.save(f'{args.input}/{scene_token}.npy', np.array(input))
            data['scenes'].append({'input': f'{scene_token}.npy', 'tokens':tokens})
        else:
            data['scenes'].append({'tokens':tokens})
    with open(f'{args.input}/tokens.json', "w") as json_file:
        json.dump(data, json_file, indent=4)
    with open(f'{args.input}/nusc_tiny_dataset.json', "w") as json_file:
        json.dump(nusc_tiny_dataset, json_file, indent=4)
