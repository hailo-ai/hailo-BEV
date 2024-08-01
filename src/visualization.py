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
COLOR_BLACK = (0, 0, 0)

LINE_WIDTH = 2
FRAME_WIDTH = 60
FRAME_HEIGHT = 60
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 338
TOP_IMAGE_SIZE = (640, 640, 3)
NUM_OF_IMAGES = 6

def render_cv2_top_view(box,
                   im: np.ndarray,
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
            cv2.line(im,
                        (int(prev[0]), int(prev[1])),
                        (int(corner[0]), int(corner[1])),
                        color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                    (int(corners.T[i][0]), int(corners.T[i][1])),
                    (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                    colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
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
    elif "vehicle.car" in category_name or "vehicle.truck" in category_name or "vehicle.construction" in category_name:
        c = COLOR_YELLOW
    elif "vehicle.bicycle" in category_name or "vehicle.motorcycle" in category_name:
        c = COLOR_DARK_BLUE
    else:
        c = COLOR_BLACK
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
    Rotates an image by 90 degrees
    """
    rotation_matrix = cv2.getRotationMatrix2D((int(size/2), int(size/2)), degrees, 1)
    rotated_image = cv2.warpAffine(bev_image, rotation_matrix, (size, size))
    return rotated_image

def show_bev(bev_image, annos, pose_record) -> np.ndarray:
    """
    Renders annotated bounding boxes on a top view radar image.

    Parameters:
    - bev_image (np.ndarray): Background image in numpy array format.
    - annos (list): List of annotations, each annotation containing 'translation', 'size', 
    'rotation', 'category_name', and 'token' keys.
    - pose_record (dict): Dictionary containing ego pose information with 'translation' and
    'rotation' keys.

    Returns:
    - np.ndarray: Image with rendered bounding boxes, rotated 90 degrees clockwise.
    """
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

def show_image_with_boxes(cv2_image, token, annos, nusc) -> None:
    """
    Renders annotated bounding boxes on a camera image using NuScenes dataset annotations.

    Parameters:
    - cv2_image (np.ndarray): Camera image in OpenCV format (BGR color order).
    - token (str): Sample token identifying the specific camera image.
    - annos (list): List of annotations, each annotation containing 'translation',
    'size', 'rotation', 'category_name', and 'token' keys.
    - nusc (NuScenes): NuScenes dataset object containing information about ego pose
    and calibrated sensors.

    Returns:
    - None
    """
    sd_record = nusc.get('sample_data', token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
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

def combine_images(images, bev_image) -> np.ndarray:
    """
    Combines a set of 6 camera images and a bird's-eye view image into a single composite image.

    Parameters:
    - images (list of np.ndarray): List containing 6 camera images in OpenCV format
    (BGR color order).
    - bev_image (np.ndarray): Bird's-eye view image in numpy array format.

    Returns:
    - np.ndarray: Combined image with 6 camera views arranged in a 2x3 grid
    and bird's-eye view image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 0)
    # Ensure that there are exactly 6 images
    if len(images) != NUM_OF_IMAGES:
        raise ValueError("Please provide exactly 6 images.")

    image_dimensions = [img.size for img in images]
    if len(set(image_dimensions)) > 1:
        print("Error: All images must have the same dimensions.")
        return

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
            y_offset = i * (img_height + FRAME_HEIGHT) +FRAME_HEIGHT
            img_np = resized_images[index]
            combined_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width, :] = img_np

            # Draw text directly on combined_image
            text = f'CAM {["FRONT LEFT", "FRONT", "FRONT RIGHT", "BACK LEFT", "BACK", "BACK RIGHT"][index]}'
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x_offset + (img_width - text_size[0]) // 2
            text_y = y_offset - 10  # Place text 10 pixels above the image
            cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Paste bev_image onto combined_image
    bev_height, bev_width, _ = bev_image.shape
    x_offset = total_width - bev_width
    y_offset = int((total_height - bev_height)/2)

    combined_image[y_offset:y_offset+bev_height, x_offset:x_offset+bev_width, :] = bev_image
    combined_image[y_offset, x_offset:, :] = [255, 255, 255]

    return combined_image

def render_ego_centric_map(nusc,
                           sample_data_token: str,
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
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
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

def viz_proc(in_queue, iterations_num, fps_calc, pose_record, infinite_loop, cam_images, tokens, nusc) -> None:
    """
    Visualizes a sequence of camera images with annotated bounding boxes and optionally
    a top-down radar view.

    Parameters:
    - in_queue (queue.Queue): Queue from which annotations are retrieved for each iteration.
    - iterations_num (int): Number of iterations or frames to process.
    - show_top_view (bool): Flag indicating whether to display a top-down radar view overlay.
    - pose_record (list): List of pose records corresponding to each frame.
    - infinite_loop (bool): Flag indicating whether to continuously loop through iterations.
    - cam_images (list of lists of np.ndarray): List of lists containing camera images
    for each camera view.
    - tokens (list of lists of str): List of lists containing tokens identifying each camera image.
    - nusc (NuScenes): NuScenes dataset object containing information about annotations
    and camera intrinsic parameters.

    Returns:
    - None
    """
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
    while True:
        for i in range(iterations_num):
            annos = in_queue.get()

            bev_image = render_ego_centric_map(nusc, tokens[NUM_OF_IMAGES][i])
            cv2.rectangle(bev_image, (int(TOP_IMAGE_SIZE[0]/2) - 20,
                                        int(TOP_IMAGE_SIZE[0]/2) - 10),
                                        (int(TOP_IMAGE_SIZE[0]/2) + 20,
                                        int(TOP_IMAGE_SIZE[0]/2) + 10),
                                        COLOR_GREEN, thickness=LINE_WIDTH)

            top_image = show_bev(bev_image,annos,pose_record[i])
            current_images = []
            j= 0
            for cam_image, token in zip(cam_images, tokens):
                if j < NUM_OF_IMAGES:
                    show_image_with_boxes(cam_image[i], token[i], annos, nusc)
                    current_images.append(cam_image[i])
                j = j + 1

            combine_img = combine_images(current_images, top_image)
            frame = cv2.cvtColor(np.array(combine_img), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', frame)
            cv2.waitKey(1)  # Wait indefinitely for a key press
            fps_calc.update_fps()
        if not infinite_loop:
            break
    cv2.destroyAllWindows()
