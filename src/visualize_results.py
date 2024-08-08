#!/usr/bin/env python3
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=C0301
# pylint: disable=W0621
# pylint: disable=W1514
# pylint: disable=R0912
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=R0915
# pylint: disable=R1702

import cv2
import numpy as np
import os
import os.path as osp
import json
from typing import Tuple, List, Union
import visualization
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

def get_boxes(sample_data_token: str, scene_annos, nusc) -> List[Box]:
    """
    Instantiates Boxes for all annotation for a particular sample_data record.
    If the sample_data is a keyframe, this returns the annotations for that sample. 
    But if the sample_data is an intermediate sample_data, a linear interpolation is applied 
    to estimate the location of the boxes at the time the sample_data was captured.
    :param sample_data_token: Unique sample_data identifier.
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = sd_record['sample_token']
    annos = scene_annos[sample]
    boxes = []
    for anno in annos:
        boxes.append(Box(anno['translation'], anno['size'], Quaternion(anno['rotation']),
                         name=anno['category_name'], token=anno['token']))
    return boxes

def get_sample_data(sample_data_token: str, scene_annos, nusc,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY) -> Tuple[str, List[Box], Union[np.ndarray, None]]:
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_ann tokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego
    :frame which is aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    boxes = get_boxes(sample_data_token, scene_annos, nusc)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if sensor_record['modality'] != 'camera':
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def render_scene(
                    scene_token: str,
                    scene_annos,
                    nusc,
                    freq: float = 7,
                    imsize: Tuple[float, float] = (640, 360),
                    out_path: str = None) -> None:
    """
    Renders a full scene with all camera channels.
    :param scene_token: Unique identifier of scene to render.
    :param freq: Display frequency (Hz).
    :param imsize: Size of image to render. The larger the slower this will run.
    :param out_path: Optional path to write a video file of the rendered frames.
    """

    assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."

    if out_path is not None:
        assert osp.splitext(out_path)[-1] == '.avi'

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    first_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    last_sample_rec = nusc.get('sample', scene_rec['last_sample_token'])

    # Set some display parameters.
    layout = {
        'CAM_FRONT_LEFT': (0, 0),
        'CAM_FRONT': (imsize[0], 0),
        'CAM_FRONT_RIGHT': (2 * imsize[0], 0),
        'CAM_BACK_LEFT': (0, imsize[1]),
        'CAM_BACK': (imsize[0], imsize[1]),
        'CAM_BACK_RIGHT': (2 * imsize[0], imsize[1]),
        'LIDAR_TOP': (3 * imsize[0], 0),
    }

    horizontal_flip = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    time_step = 1 / freq * 1e6  # Time-stamps are measured in micro-seconds.

    canvas = np.ones((2 * imsize[1], 4 * imsize[0], 3), np.uint8)
    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_path, fourcc, freq, canvas.shape[1::-1])
    else:
        out = None

    # Load first sample_data record for each channel.
    current_recs = {}  # Holds the current record to be displayed by channel.
    prev_recs = {}  # Hold the previous displayed record by channel.
    for channel in layout:
        current_recs[channel] = nusc.get('sample_data', first_sample_rec['data'][channel])
        prev_recs[channel] = None

    current_time = first_sample_rec['timestamp']
    while current_time < last_sample_rec['timestamp']:

        current_time += time_step

        # For each channel, find first sample that has time > current_time.
        for channel, sd_rec in current_recs.items():
            while sd_rec['timestamp'] < current_time and sd_rec['next'] != '':
                sd_rec = nusc.get('sample_data', sd_rec['next'])
                current_recs[channel] = sd_rec

        for channel, sd_rec in current_recs.items():
            if not sd_rec == prev_recs[channel]:
                impath, boxes, camera_intrinsic = get_sample_data(sd_rec['token'], scene_annos, nusc,
                                                    box_vis_level=BoxVisibility.ANY)
                if channel == "LIDAR_TOP":
                    im = visualization.render_ego_centric_map(nusc, current_recs[channel]['token'])
                    cv2.rectangle(im, (300, 310), (340, 330), (0, 255, 0), thickness=2)
                    for box in boxes:
                        c = visualization.get_color(box.name)
                        if "movable_object" not in box.name:
                            visualization.render_cv2_top_view(box, im, view=np.eye(4), colors=(c, c, c))

                    top_image = visualization.rotate_image(im, 90, 640)
                    top_image = cv2.cvtColor(np.array(top_image), cv2.COLOR_RGB2BGR)
                    top_image = cv2.resize(top_image, (imsize[0], 2 * imsize[1]))
                    canvas[
                        layout[channel][1]: layout[channel][1] + 2 * imsize[1],
                        layout[channel][0]: layout[channel][0] + imsize[0], :
                    ] = top_image

                else:
                    im = cv2.imread(impath)
                    for box in boxes:
                        c = visualization.get_color(box.name)
                        c = visualization.rgb_2_bgr(c)
                        if "movable_object" not in box.name:
                            box.render_cv2(im, view=camera_intrinsic, normalize=True,
                                        colors=(c, c, c))

                    im = cv2.resize(im, imsize)
                    if channel in horizontal_flip:
                        im = im[:, ::-1, :]

                    # Store here so we don't render the same image twice.
                    prev_recs[channel] = sd_rec
                    canvas[
                        layout[channel][1]: layout[channel][1] + imsize[1],
                        layout[channel][0]: layout[channel][0] + imsize[0], :
                    ] = im

        # Show updated canvas.
        cv2.imshow('BEV', canvas)
        if out_path is not None:
            out.write(canvas)

        key = cv2.waitKey(1)  # Wait a very short time (1 ms).

        if key == 32:  # if space is pressed, pause.
            key = cv2.waitKey()

        if key == 27:  # if ESC is pressed, exit.
            cv2.destroyAllWindows()
            break

def get_scene_number(scene_name):
    """
    Extracts the scene number from a scene name formatted as 'scene-XXXX'.
    """
    scene_number = int(scene_name[6:])
    return scene_number

def get_scene_token_from_name(scene_name, nusc):
    """
    Retrieves the scene token for a given scene name from a NuSC object.
    """
    scenes = nusc.scene
    scene = scenes[get_scene_number(scene_name) - 1]
    scene_token = scene['token']
    return scene_token

def get_scene_tokens_from_names(scene_names, nusc):
    """
    Retrieves scene tokens for a list of scene names from a NuSC object.
    """
    scene_tokens = []
    for scene_name in scene_names:
        scene_tokens.append(get_scene_token_from_name(scene_name, nusc))
    return scene_tokens

def render_scenes(scene_names, tokens, dir_path, nusc):
    """
    Renders scenes based on results and tokens.

    Args:
        results (List[str]): List of result identifiers (assumed to be filenames without extension).
        tokens (List[str]): List of tokens corresponding to each result.
        dir_path (str): Directory path where JSON annotation files are located.
        nusc (object): Object or data structure required for rendering scenes.

    Renders scenes based on the results and tokens provided. Each result corresponds
    to a JSON file in `dir_path`, which contains scene annotations. The function
    opens each JSON file, loads its contents as scene annotations, and renders the
    scene using `render_scene` function.
    """
    window_name = 'BEV'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    for scene_name, token in zip(scene_names, tokens):
        with open(f'{dir_path}/{scene_name}.json', 'r') as f:
            scene_annos= json.load(f)
        render_scene(token, scene_annos, nusc)

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(description="Bev visualization")
    parser.add_argument("-f", "--file", default='resources/results/scenes_data.json', required=False,
                        help="scene data file path")
    parser.add_argument("-d", "--data", default="resources/data/",
                        help="path to the data folder, where the nuScenes dataset is.")
    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.data, verbose=False)
    dir_path = os.path.dirname(args.file)

    with open(args.file, 'r') as f:
        scene_data = json.load(f)
        scene_names = scene_data['scenes']
        tokens  = get_scene_tokens_from_names(scene_names, nusc)

        render_scenes(scene_names, tokens, dir_path, nusc)
