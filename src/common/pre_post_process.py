# pylint: disable=C0301
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=R1735
import os
import queue
import onnxruntime
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box
import cv2

def preprocess(data_path, files):
    images_after_pre = []
    for filename in files:
        img = cv2.imread(data_path + filename)
        img = cv2.resize(img, (800, 450))
        x, y, width, height = 0, 130, 800, 450  
        img = img[y:y + height, x:x + width]

        images_after_pre.append(img)
    return np.array(images_after_pre)

def post_proc(in_queue, in_meta_queue, out_queue, post_proc_onnx_path, demo) -> None:
    """
    Perform post-processing on input data using an ONNX model
    and put the results into an output queue.
    Args:
    - in_queue (Queue object): Queue containing input data to be processed.
    - out_queue (Queue object): Queue where processed results are put.
    - iterations_num (int): Number of iterations to process data.
    - infinite_loop (bool): Flag indicating whether to loop indefinitely.
    - post_proc_onnx_path (str): Path to the ONNX model for post-processing.
    - timestamp (list): List of numpy arrays representing timestamps corresponding to input data.

    Returns:
    - None

    Description:
    This function continuously retrieves input data from 'in_queue', performs post-processing
    using an ONNX model loaded from 'post_proc_onnx_path', and places the processed results
    into 'out_queue'. Each iteration processes data from 'in_queue' along with the respective
    timestamp from the 'timestamp' list.
    """
    assert os.path.exists(post_proc_onnx_path), f'File not found: {post_proc_onnx_path}'
    sess_post_proc = onnxruntime.InferenceSession(post_proc_onnx_path)

    timest = np.array([1.25]).astype(np.float32)

    output_names = [x.name for x in sess_post_proc.get_outputs()]

    while True:
        while not demo.get_terminate():
            try:
                transformer_output = in_queue.get(block=True, timeout=1)
                break
            except queue.Empty:
                pass
        while not demo.get_terminate():
            try:
                meta_data = in_meta_queue.get(block=True, timeout=0.5)
                break
            except queue.Empty:
                pass
        if demo.get_terminate():
            break

        assert transformer_output['petrv2_repvggB0_transformer_pp_800x320/concat1'][0].shape == (1, 304, 10), "Expected shape of matmul is (1, 304, 10), but got {}".format(transformer_output['petrv2_repvggB0_transformer_pp_800x320/concat1'][0].shape)
        pp_in = {
                '4062'  : transformer_output['petrv2_repvggB0_transformer_pp_800x320/concat1'][0],
                '4551' :  np.expand_dims(np.expand_dims(timest, axis=0), axis=0)
        }

        results = sess_post_proc.run(output_names, pp_in)
        result = np.stack((np.array(transformer_output['petrv2_repvggB0_transformer_pp_800x320/conv41']), np.array(results[0])),axis=0)

        while not demo.get_terminate():
            try:
                out_queue.put((result, meta_data), block=True, timeout=0.5)
                break
            except queue.Full:
                pass

        if demo.get_terminate():
            break

def denormalize_bbox(normalized_bboxes) -> torch.Tensor:
    """
    Denormalize bounding boxes from their normalized representation.

    Args:
    - normalized_bboxes (torch.Tensor): Normalized bounding boxes with shape (..., N),
    where N >= 7. The last two elements (if present) represent additional parameters
    like velocity.

    Returns:
    - torch.Tensor: Denormalized bounding boxes with shape (..., M), where M = N if
    N <= 8, otherwise M = N + 2.

    Description:
    This function takes normalized bounding boxes and performs denormalization:
    - Calculates rotation angle based on sine and cosine values.
    - Retrieves center coordinates (cx, cy, cz) in the bird's-eye view (BEV).
    - Computes size dimensions (width, length, height) by exponentiating corresponding
    dimensions in the input.
    - If additional velocity parameters (vx, vy) are present in the input, they are
    concatenated with the denormalized results.
    - Returns denormalized bounding boxes suitable for further processing or visualization.
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

def decode_single(cls_scores, bbox_preds) -> dict:
    """
    Decode and post-process predicted bounding boxes and classification scores.

    Args:
    - cls_scores (torch.Tensor): Tensor containing classification scores.
    - bbox_preds (torch.Tensor): Tensor containing predicted bounding box parameters.

    Returns:
    - dict: Dictionary containing decoded predictions with keys 'bboxes', 'scores', 'labels'.

    Description:
    This function decodes the predicted classification scores and bounding box parameters:
    - Applies sigmoid activation to convert classification scores.
    - Top-k operation to select up to 'max_num' predictions based on scores.
    - Maps indices to labels and corresponding bounding box predictions.
    - Denormalizes bounding box predictions using 'denormalize_bbox' function.
    - Filters predictions based on 'post_center_range' to ensure bounding boxes are within
    specified limits.
    - Returns a dictionary with decoded bounding boxes ('bboxes'), scores ('scores'), and
    labels ('labels').
    """
    max_num = 30
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
    cls_scores = cls_scores.sigmoid()
    scores, indexs = cls_scores.view(-1).topk(max_num)
    labels = indexs % 10
    bbox_index = indexs // 10
    bbox_preds = bbox_preds[bbox_index]
    final_box_preds = denormalize_bbox(bbox_preds)
    final_scores = scores
    final_preds = labels
    predictions_dict = {}
    if post_center_range is not None:
        post_center_range = torch.tensor(post_center_range, device=scores.device)
        mask = (final_box_preds[..., :3] >=
                post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <=
                    post_center_range[3:]).all(1)
        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]
        labels = final_preds[mask]
        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
    return predictions_dict

def bbox3d2result(bboxes, scores, labels, attrs=None) -> dict:
    """
    Convert 3D bounding boxes, scores, labels, and optional attributes into a dictionary format.

    Args:
        bboxes (list): List of 3D bounding boxes.
        scores (list): List of scores corresponding to each 3D bounding box.
        labels (list): List of labels corresponding to each 3D bounding box.
        attrs (dict, optional): Optional dictionary of additional attributes for each 3D bounding box.

    Returns:
        dict: Dictionary containing the following keys:
            - 'boxes_3d': List of 3D bounding boxes (`bboxes`).
            - 'scores_3d': List of scores (`scores`) corresponding to each 3D bounding box.
            - 'labels_3d': List of labels (`labels`) corresponding to each 3D bounding box.
            - 'attrs_3d' (optional): Dictionary of additional attributes (`attrs`) if provided.
    """
    result_dict = dict(
        boxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels)

    if attrs is not None:
        result_dict['attrs_3d'] = attrs

    return result_dict

def decode(outs) -> dict:
    """
    Decode and process predictions from model outputs.

    Args:
    - outs (list): List containing model outputs. Expected structure is [cls_scores, bbox_preds],
    where cls_scores and bbox_preds are lists or tensors of classification scores and bounding
    box predictions respectively.

    Returns:
    - dict: Dictionary containing decoded predictions with key 'pts_bbox'.

    Description:
    This function decodes predictions from model outputs:
    - Extracts classification scores and bounding box predictions from 'outs'.
    - Iterates over each sample in the batch to decode predictions using 'decode_single'.
    - Adjusts bounding box dimensions based on computed parameters.
    - Constructs a list of bounding box results using 'bbox3d2result'.
    - Returns a dictionary with decoded bounding box results under 'pts_bbox'.
    """
    all_cls_scores = torch.tensor(outs[0][-1])
    all_bbox_preds = torch.tensor(outs[1][-1])
    batch_size = 1
    predictions_list = []
    for i in range(batch_size):
        predictions_list.append(decode_single(all_cls_scores[i], all_bbox_preds[i]))
    num_samples = len(predictions_list)
    bbox_list = []
    for i in range(num_samples):
        preds = predictions_list[i]
        bboxes = preds['bboxes'].numpy()
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        scores = preds['scores'].numpy()
        labels = preds['labels'].numpy()
        bbox_list.append([bboxes, scores, labels])
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]

    result_dict = {}
    result_dict['pts_bbox'] = bbox_results
    return result_dict

def gravity_center(box3d) -> torch.Tensor:
    """torch.Tensor: A tensor with center of each box."""
    bottom_center = box3d[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, :2] = bottom_center[:, :2]
    gravity_center[:, 2] = bottom_center[:, 2] + box3d[:, 5] * 0.5
    return gravity_center

def output_to_nusc_box(detection) -> list:
    """Convert the output to the box class in the nuScenes.

    Args:
    detection (dict): Detection results.
    - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
    - scores_3d (torch.Tensor): Detection scores.
    - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: list of standard NuScenesBoxes.
    """
    box3d = detection['pts_bbox'][0]['boxes_3d']
    scores = detection['pts_bbox'][0]['scores_3d']
    labels = detection['pts_bbox'][0]['labels_3d']

    box_gravity_center = gravity_center(box3d)
    box_dims = box3d[:, 3:6]
    box_yaw = box3d[:, 6]
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d[i, 7:9], 0.0)

        box = Box(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(sample_token,
                            boxes,
                            classes, nusc) -> list:
    """
    Transform LiDAR-detected bounding boxes to global coordinates in NuScenes dataset.

    Args:
    - sample_token (str): Token of the sample in NuScenes dataset.
    - boxes (list): List of Box objects representing LiDAR-detected bounding boxes.
    - classes (dict): Dictionary mapping class labels to detection ranges.
    - nusc (NuScenes): NuScenes database instance.

    Returns:
    - list: List of Box objects transformed to global coordinates.

    Description:
    This function transforms LiDAR-detected bounding boxes to global coordinates:
    - Retrieves sample, sensor data, calibrated sensor, and ego pose information from NuScenes.
    - Iterates over each box in 'boxes' and rotates/translates it according to calibrated sensor
    and ego pose transformations.
    - Filters boxes based on their detection range defined in 'classes'.
    - Returns a list of Box objects transformed to global coordinates.
    """
    box_list = []
    pose_record = nusc[sample_token]['LIDAR_TOP'][0]
    cs_record = nusc[sample_token]['LIDAR_TOP'][1]

    for box in boxes:
        box.rotate(pyquaternion.Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        cls_range_map = {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50,
                        'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40,
                        'bicycle': 40, 'traffic_cone': 30, 'barrier': 30}
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        box.rotate(pyquaternion.Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list

def _format_bbox(results, sample, nusc) -> str:
    """Convert the results to the standard format.

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
        You can specify the output directory/filename by
        modifying the jsonfile_prefix. Default: None.

    Returns:
        str: Path of the output json file.
    """
    mapped_class_names = ['car',
                        'truck',
                        'construction_vehicle',
                        'bus',
                        'trailer',
                        'barrier',
                        'motorcycle',
                        'bicycle',
                        'pedestrian',
                        'traffic_cone']
    default_attribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    category_mapping = {'barrier': 'movable_object.barrier',
                    'bicycle': 'vehicle.bicycle',
                    'bus': 'vehicle.bus.rigid',
                    'car': 'vehicle.car',
                    'construction_vehicle': 'vehicle.construction',
                    'motorcycle': 'vehicle.motorcycle',
                    'pedestrian': 'human.pedestrian.police_officer',
                    'traffic_cone': 'movable_object.trafficcone',
                    'trailer': 'vehicle.trailer', 'truck': 'vehicle.truck'}
    det = results
    annos = []
    bboxes= []
    boxes = output_to_nusc_box(det)

    boxes = lidar_nusc_box_to_global(sample, boxes, mapped_class_names, nusc)

    for _, box in enumerate(boxes):
        name = mapped_class_names[box.label]
        if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
            if name in [
                    'car',
                    'construction_vehicle',
                    'bus',
                    'truck',
                    'trailer',
            ]:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = default_attribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = default_attribute[name]

        nusc_anno = dict(
            sample_token=sample,
            translation=box.center.tolist(),
            size=box.wlh.tolist(),
            rotation=box.orientation.elements.tolist(),
            velocity=box.velocity[:2].tolist(),
            detection_name=name,
            detection_score=box.score,
            attribute_nasme=attr)
        annos.append(nusc_anno)
        bboxes.append(box)
    result_annos = []
    token = 0
    for record in annos:
        if record["detection_score"] > 0.25 and record['detection_name'] in category_mapping:
            record["token"] = str(token)
            record['category_name'] = category_mapping[record['detection_name']]
            result_annos.append(record)
            token = token + 1
    return result_annos

def d3nms_proc(in_queue, out_queue, nusc, demo) -> None:
    """
    Perform 3D Non-Maximum Suppression (NMS) and annotation formatting for NuScenes dataset.

    Args:
    - in_queue (Queue): Queue containing input data for processing.
    - out_queue (Queue): Queue to put processed annotations.
    - tok_queue (Queue): Queue containing tokens for processing.
    - iterations_num (int): Number of iterations to process.
    - infinite_loop (bool): Flag indicating whether to loop infinitely.
    - nusc (NuScenes): NuScenes database instance.

    Returns:
    - None

    Description:
    This function performs 3D Non-Maximum Suppression (NMS) and formats annotations:
    - Retrieves preprocessed output from 'in_queue'.
    - Retrieves token from 'tok_queue' for annotation formatting.
    - Decodes bounding boxes from preprocessed output using 'decode' function.
    - Formats decoded bounding boxes and tokens into annotations using '_format_bbox'.
    - Puts formatted annotations into 'out_queue'.
    """

    while True:
        while not demo.get_terminate():
            try:
                pp_output, meta_data = in_queue.get(block=True, timeout=0.5)
                break
            except queue.Empty:
                pass

        if demo.get_terminate():
            break

        assert pp_output.shape == (2, 1, 1, 304, 10),  "Expected shape of Post-process output is (2, 1, 1, 304, 10), but got {}".format(pp_output.shape)
        bbox_list = decode(pp_output)
        annos = _format_bbox(bbox_list, meta_data, nusc)
        while not demo.get_terminate():
            try:
                out_queue.put((annos, meta_data), block=True, timeout=0.5)
                break
            except queue.Full:
                pass

        if demo.get_terminate():
            break
