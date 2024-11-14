# pylint: disable=C0301
# pylint: disable=E1101
# pylint: disable=C0411
# pylint: disable=R0914
# pylint: disable=R0913
# pylint: disable=C0200

from pyquaternion import Quaternion
import numpy as np
import cv2
import math
from typing import Tuple

from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.utils.data_classes import Box

# Constants
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (128, 0, 0)
COLOR_GRAY = (155, 155, 155)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 158, 255)
COLOR_DARK_BLUE = (60, 20, 220)
COLOR_ORANGE = (0, 80, 255)

LINE_WIDTH = 2
FRAME_WIDTH = 60
FRAME_HEIGHT = 60
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 338
TOP_IMAGE_SIZE = (640, 640, 3)
NUM_OF_IMAGES = 6


def render_ego_centric_map(nusc,
                           sample_token: str,
                           axes_limit: float = 40) -> None:
    """
    Render map centered around the associated ego pose.
    :param sample_data_token: Sample_data token.
    :param axes_limit: Axes limit measured in meters.
    :param ax: Axes onto which to render.
    """

    def crop_image(image: np.array,
                    x_px: int,
                    y_px: int,
                    axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    # Get data.
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_image = rotate_image(cropped, yaw_deg, cropped.shape[0])

    # Crop image.
    ego_centric_map = crop_image(rotated_image,
                                    int(rotated_image.shape[1] / 2),
                                    int(rotated_image.shape[0] / 2),
                                    scaled_limit_px)
    # Change colors.
    ego_centric_map[ego_centric_map == map_mask.background] = 254
    ego_centric_map[ego_centric_map == map_mask.foreground] = 125

    # Resize image.
    resized_image = cv2.resize(ego_centric_map, (640, 640))
    result_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    return result_image

def render_cv2_top_view(box,
                   img: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = (COLOR_RED, COLOR_BLUE, COLOR_GRAY),
                   linewidth: int = LINE_WIDTH) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
    :param linewidth: Linewidth for plot.
    """

    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
    corners[1, :] *= -1
    for i in range(len(corners)):
        for j in range(len(corners[i])):
            corners[i][j] += 40
            corners[i][j] *= 8

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                        (int(prev[0]), int(prev[1])),
                        (int(corner[0]), int(corner[1])),
                        color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(img,
                    (int(corners.T[i][0]), int(corners.T[i][1])),
                    (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                    colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(img,
                (int(center_bottom[0]), int(center_bottom[1])),
                (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                colors[0][::-1], linewidth)

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories,
    as well as the nuScenes detection categories.
    """
    if "human" in category_name:
        c = COLOR_BLUE
    elif "vehicle.car" in category_name or "vehicle.construction" in category_name:
        c = COLOR_YELLOW
    elif "vehicle.bicycle" in category_name or "vehicle.motorcycle" in category_name:
        c = COLOR_DARK_BLUE
    else:
        c = COLOR_ORANGE
    return c

def rgb_2_bgr(color) -> Tuple[int, int, int]:
    """
    Converts rgb color to bgr color
    """
    r = color[0]
    g = color[1]
    b = color[2]
    return (b, g, r)

def rotate_image(bev_image, degrees, size) -> np.ndarray:
    """
    Rotates an image by degrees
    """
    rotation_matrix = cv2.getRotationMatrix2D((int(size/2), int(size/2)), degrees, 1)
    rotated_image = cv2.warpAffine(bev_image, rotation_matrix, (size, size))
    return rotated_image

def show_bev(bev_image, annos, sample_token, nusc) -> np.ndarray:
    """
    Renders annotated bounding boxes on a top-view radar (bird's-eye view) image using NuScenes dataset annotations.

    Parameters:
    - bev_image (np.ndarray): Background bird's-eye view image (e.g., from radar or lidar) in numpy array format.
    - annos (list of dict): List of annotation dictionaries for the objects to be rendered.
      Each annotation contains:
        - 'translation' (list): Position of the object in 3D space.
        - 'size' (list): Dimensions of the bounding box for the object.
        - 'rotation' (list): Quaternion representing the object's rotation.
        - 'category_name' (str): Name of the object category.
        - 'token' (str): Unique identifier for the annotation.
    - sample_token (str): Unique identifier for the specific sample in the NuScenes dataset.
    - nusc (NuScenes): NuScenes dataset instance providing access to dataset objects and metadata.

    Returns:
    - np.ndarray: The annotated bird's-eye view image, rotated 90 degrees clockwise for correct orientation.
    """
    pose_record = nusc[sample_token]['LIDAR_TOP'][0]
    boxes = []
    for anno in annos:
        boxes.append(Box(anno['translation'], anno['size'], Quaternion(anno['rotation']),
                        name=anno['category_name'], token=anno['token']))
    box_list = []
    for box in boxes:
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        box_list.append(box)
        c = get_color(box.name)
        if "movable_object" not in box.name:
            render_cv2_top_view(box, bev_image, view=np.eye(4), colors=(c, c, c))
    rotated_image = rotate_image(bev_image, 90, bev_image.shape[0])
    return rotated_image

def show_image_with_boxes(sample_token, cv2_image, cam, annos, nusc) -> None:
    """
    Renders annotated bounding boxes on a camera image using NuScenes dataset annotations.

    Parameters:
    - sample_token (str): Unique identifier for the specific sample in the NuScenes dataset.
    - cv2_image (np.ndarray): Camera image in OpenCV format (BGR color order) on which the
      bounding boxes will be drawn.
    - cam (str): Camera channel name (e.g., 'CAM_FRONT') indicating which camera's data to use.
    - annos (list of dict): List of annotation dictionaries for the objects to be rendered.
      Each annotation contains:
        - 'translation' (list): Position of the object in 3D space.
        - 'size' (list): Dimensions of the bounding box for the object.
        - 'rotation' (list): Quaternion representing the object's rotation.
        - 'category_name' (str): Name of the object category.
        - 'token' (str): Unique identifier for the annotation.
    - nusc (NuScenes): NuScenes dataset instance providing access to dataset objects and metadata.

    Returns:
    - None: The function modifies the `cv2_image` in-place by drawing the bounding boxes,
      without returning any value.
    # """

    pose_record = nusc[sample_token][cam][0]
    cs_record = nusc[sample_token][cam][1]

    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    imsize = (1600, 900)
    boxes = []
    for anno in annos:
        boxes.append(Box(anno['translation'], anno['size'], Quaternion(anno['rotation']),
                        name=anno['category_name'], token=anno['token']))
    box_list = []

    for box in boxes:
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        if not box_in_image(box, cam_intrinsic, imsize, vis_level = BoxVisibility.ANY):
            continue
        box_list.append(box)

    for box in box_list:
        c = get_color(box.name)
        if "movable_object" not in box.name:
            box.render_cv2(cv2_image, view=cam_intrinsic, normalize=True, colors=(c, c, c))

def combine_images(images, bev_image, fps_calc) -> np.ndarray:
    """
    Combines a set of 6 camera images and a bird's-eye view (BEV) image into a single composite image,
    with camera labels and FPS information.

    Parameters:
    - images (list of np.ndarray): List of 6 camera images in OpenCV format (BGR color order).
      Each image is expected to have the same dimensions.
    - bev_image (np.ndarray): Bird's-eye view (BEV) image as a numpy array in BGR format.
    - fps_calc (object): Object with a `get_fps()` method that returns the current FPS value.

    Returns:
    - np.ndarray: A single composite image containing:
        - 6 camera images arranged in a 2x3 grid, labeled by camera positions.
        - The BEV image, positioned alongside the camera images.
        - FPS information displayed at a specified position on the composite image.

    Raises:
    - ValueError: If the number of images provided is not exactly 6.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 0)
    # Ensure that there are exactly 6 images
    # assert len(images) != NUM_OF_IMAGES, f'Please provide exactly {NUM_OF_IMAGES} images.'

    image_dimensions = [img.size for img in images]
    # assert len(set(image_dimensions)) > 1, "Error: All images must have the same dimensions."

    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        resized_images.append(resized_image)

    # Assuming all images have the same dimensions, get the width and height of one image
    img_width, img_height = resized_images[0].shape[1], resized_images[0].shape[0]
    total_width = 3 * (img_width + 2 * FRAME_WIDTH) + bev_image.shape[0]
    total_height = 2 * (img_height + 2 * FRAME_HEIGHT)

    combined_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # Convert the resized images to NumPy arrays and paste onto combined_image
    for i in range(2):
        for j in range(3):
            index = i * 3 + j
            x_offset = j * (img_width + FRAME_WIDTH) + FRAME_WIDTH
            y_offset = i * (img_height + FRAME_HEIGHT) + FRAME_HEIGHT
            img_np = resized_images[index]
            combined_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width, :] = img_np

            # Draw text directly on combined_image
            text = f'CAM {["FRONT LEFT", "FRONT", "FRONT RIGHT", "BACK LEFT", "BACK", "BACK RIGHT"][index]}'
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x_offset + (img_width - text_size[0]) // 2
            text_y = y_offset - 10  # Place text 10 pixels above the image
            cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    bev_height, bev_width, _ = bev_image.shape
    x_offset = total_width - bev_width
    y_offset = int((total_height - bev_height)/2)
    float_fps = float(fps_calc.get_fps())
    if float_fps > 14:
        float_fps = f"{float_fps:.2f}"
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = 2320
        text_y = y_offset - 83
        cv2.putText(combined_image, f'FPS {float_fps}', (text_x, text_y), font, font_scale, font_color, font_thickness)
    combined_image[y_offset:y_offset+bev_height, x_offset:x_offset+bev_width, :] = bev_image
    combined_image[y_offset, x_offset:, :] = [255, 255, 255]

    return combined_image

def viz_proc(input_path, data_path, in_queue, fps_calc, nusc) -> None:
    """
    Visualization process that handles displaying images, annotations, and 3D bounding boxes in a window.

    This function runs in an infinite loop, fetching camera images and annotations for a given token from the
    `in_queue`, and renders these images with bounding boxes and other visualizations in a window. It also handles
    FPS calculation and updating.

    Args:
        in_queue (Queue): A queue that provides camera images and annotation data (in the form of a tuple `(annos, token)`).
        fps_calc (fps_calc.FPSCalc): An instance of the `FPSCalc` class used to calculate and update FPS.
        nusc (NuScenes): An instance of the `NuScenes` class to fetch data from the nuScenes dataset (camera images, annotations, etc.).

    Returns:
        None: The function runs indefinitely, continuously processing and displaying visualizations.
    """
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    while True:
        cam_images = []
        annos, token = in_queue.get()
        cams = ['CAM_FRONT_LEFT','CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        for cam in cams:
            img = cv2.imread(data_path + nusc[token][cam][2]['filename'])
            cam_images.append(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        bev_image = cv2.imread(input_path + nusc[token]['LIDAR_TOP'][2]['filename'])

        cv2.rectangle(bev_image, (int(TOP_IMAGE_SIZE[0]/2) - 20,
                                    int(TOP_IMAGE_SIZE[0]/2) - 10),
                                    (int(TOP_IMAGE_SIZE[0]/2) + 20,
                                    int(TOP_IMAGE_SIZE[0]/2) + 10),
                                    COLOR_GREEN, thickness=LINE_WIDTH)

        top_image = show_bev(bev_image,annos,token, nusc)
        current_images = []
        j= 0

        for cam_image, cam in zip(cam_images, cams):
            if j < NUM_OF_IMAGES:
                show_image_with_boxes(token, cam_image, cam, annos, nusc)
                current_images.append(cam_image)
            j = j + 1

        combine_img = combine_images(current_images, top_image, fps_calc)
        frame = cv2.cvtColor(np.array(combine_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)  # Wait indefinitely for a key press
        fps_calc.update_fps()
