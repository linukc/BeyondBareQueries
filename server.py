import os
import time
from copy import deepcopy
import textwrap
import gzip
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import sys
sys.path.append("/home/jovyan/Tatiana_Z/bbq_demo/MobileSAM/MobileSAMv2")

import cv2
import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor
from bbq.models import LLaVaChat
from bbq.grounding import Llama3

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

# Путь к файлу, куда клиент отправляет текст
TEXT_FILE = "/home/jovyan/Tatiana_Z/bbq_demo/user_query/text.txt"
COLOR_PATH = "/home/jovyan/Tatiana_Z/bbq_demo/user_query/image.png"
DEPTH_PATH = "/home/jovyan/Tatiana_Z/bbq_demo/user_query/depth.png"
POSE_PATH = "/home/jovyan/Tatiana_Z/bbq_demo/user_query/pose.txt"
CONFIG_FILE = "/home/jovyan/Tatiana_Z/bbq_demo/BeyondBareQueries/examples/configs/lab/room0.yaml"
SAVE_PATH = "/home/jovyan/Tatiana_Z/bbq_demo/outputs"
LLAMA_PATH = "/workspace-SR006.nfs2/Tatiana_Z/Meta-Llama-3-8B-Instruct"

DEBUG = False

INTRINSICS = np.array([
    [954.4869995117188, 0, 639.8150634765625, 0],  # fx,  0, cx, tx
    [0, 954.4869995117188, 350.4607238769531, 0],  #  0, fy, cy, ty
    [0, 0, 1, 0]       #  0,  0,  1,  0
], dtype=np.float32)
DEPTH_SCALE = 1000
FIRST_POSE = None

font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 0.75  # Font size
COLOR_CV = (0, 0, 0)  # Text color (BGR: red)
thickness = 2 # Thickness of text

hash = datetime.now()
with open(CONFIG_FILE) as file:
    config = yaml.full_load(file)

nodes_constructor = NodesConstructor(config["nodes_constructor"])

if not DEBUG:
    chat = LLaVaChat()
    logger.info("LLaVA chat is initialized.")
    llm = Llama3(LLAMA_PATH)
    logger.info("LLama3 chat is initialized.")


class TqdmLoggingHandler:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message, **kwargs):
        if message.strip() != "":
            tqdm.write(message, end="")

    def flush(self):
        pass

def set_seed(seed: int=18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def get_xyxy_from_mask(mask):
    non_zero_indices = np.nonzero(mask)

    if non_zero_indices[0].sum() == 0:
        return (0, 0, 0, 0)
    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])

    return (x_min, y_min, x_max, y_max)

def crop_image(image, mask, padding=30):
    image = np.array(image)
    x1, y1, x2, y2 = get_xyxy_from_mask(mask)

    if image.shape[:2] != mask.shape:
        logger.critical(
            "Shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape)
        )
        raise RuntimeError

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image
    image_crop = image[y1:y2, x1:x2]

    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop

def describe_objects(objects):
    result = []
    query_base = """Describe visible object in front of you, 
    paying close attention to its spatial dimensions and visual attributes."""

    for idx, object_ in tqdm(enumerate(objects)):
        template = {}

        ### spatial features
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(object_['pcd'].points))

        template["id"] = idx
        template["bbox_extent"] = [round(i, 2) for i in list(bbox.get_extent())]
        template["bbox_center"] = [round(i, 2) for i in list(bbox.get_center())]
        if not DEBUG and "description" not in object_:
            ### caption
            image = cv2.imread(COLOR_PATH) # we work with BGR images so it converts them to RGB
            image = image[..., :3] 
            image = Image.fromarray(image)

            mask = object_["local_mask"]
            image = image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            image_crop = crop_image(image, mask)
            image_features = [image_crop]
            image_sizes = [image.size for image in image_features]
            image_features = chat.preprocess_image(image_features)
            image_tensor = [image.to("cuda", dtype=torch.float16) for image in image_features]
            
            query_tail = """
            The object is one we usually see in indoor scenes, specifically in laboratories. 
            It signature must be short and sparse, describe appearance, geometry, material. Don't describe background.
            Fit you description in four or five words.
            Examples: 
            a white test tube rack;
            a robotic arm;
            a closed wooden door with a glass panel;
            a pillow with a floral pattern;
            a wooden table;
            a gray wall.
            """
            query = query_base + "\n" + query_tail
            text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
            template["description"] = text.replace("<s>", "").replace("</s>", "").strip()
        elif "description" not in object_:
            template["description"] = "Test description"
        else:
            template["description"] = object_["description"]
        print(template)
        object_["description"] = template["description"]
        result.append(template)
    return result

def project_point_cloud_to_image(image, depth_image, camera_pose, intrinsics_matrix, point_cloud):
    """
    Projects a 3D point cloud onto a 2D image using the camera pose, intrinsics, and depth image.
    
    Args:
        image (np.ndarray): The input RGB image (H x W x 3).
        camera_pose (np.ndarray): The 4x4 camera pose matrix (extrinsics).
        intrinsics_matrix (np.ndarray): The 3x3 camera intrinsics matrix.
        point_cloud (np.ndarray): The Nx3 point cloud array, where N is the number of points.
        
    Returns:
        np.ndarray: The image with the projected point cloud.
    """
    
    # Image height and width
    height, width = image.shape[:2]
    
    # Transform point cloud to camera coordinates
    # Add a column of 1s to the point cloud to make it Nx4 for matrix multiplication with pose
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud_homogeneous = np.hstack((point_cloud, ones))  # (N, 4)
    
    # Apply the camera pose to transform the points to camera coordinates
    point_cloud_camera_coords = (camera_pose @ point_cloud_homogeneous.T).T  # (N, 4)
    point_cloud_camera_coords = point_cloud_camera_coords[:, :3]  # Drop the homogeneous component

    #print(point_cloud_camera_coords)

    # Project points onto the image plane using the intrinsics matrix
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    
    # Only consider points in front of the camera (z > 0)
    #valid_points = point_cloud_camera_coords[:, 2] > 0

    #point_cloud_camera_coords = point_cloud_camera_coords[valid_points]

    # Project 3D points to 2D using the camera intrinsics
    u = (point_cloud_camera_coords[:, 0] * fx / point_cloud_camera_coords[:, 2]) + cx
    v = (point_cloud_camera_coords[:, 1] * fy / point_cloud_camera_coords[:, 2]) + cy
    
    # Round to nearest pixel and stack u, v coordinates
    pixel_coords = np.vstack((u, v)).T
    pixel_coords = np.round(pixel_coords).astype(int)

    # Filter valid pixel coordinates (those within image bounds)
    valid_pixel_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
    )
    
    #pixel_coords = pixel_coords[valid_pixel_mask]

    valid_pixel_mask = np.zeros(len(pixel_coords), dtype=bool)

    # Optionally, check against depth image to handle occlusions
    """
    for i, (u, v) in enumerate(pixel_coords):
        depth_value = depth_image[v, u]
        point_depth = point_cloud_camera_coords[i, 2]
        if np.abs(point_depth-depth_value) <= 0.05:  # Allow some tolerance for depth camera
            # Color the pixel in the image (for simplicity, color it white)
            valid_pixel_mask[i] = True
    pixel_coords = pixel_coords[valid_pixel_mask, :]
    """
    return pixel_coords


def draw_answer(result, targets, anchors, relations, segmentation, depth, intrinsics, pose, user_query, LLM_answer, save_filename, no_json=True):

    camera_pose = np.linalg.inv(pose.cpu().numpy())

    objects_by_id = {
        obj['id']: obj['bbox_center']
        for obj in result
    }
    # Example list of 3D points and labels
    points = np.array([
        obj['bbox_center'] for obj in result
        if 'A wall on the side of a building' not in obj['description'] and obj['id'] not in targets and obj['id'] not in anchors 
    ])
    labels = [f"{obj['id']}: {obj['description']}" for obj in result
             if 'A wall on the side of a building' not in obj['description'] and obj['id'] not in targets and obj['id'] not in anchors 
             ]  # Labels for each point

    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Create 3D figure
    # Create a figure with two subplots (one for 3D plot, one for text)
    fig = plt.figure(figsize=(5, 5))  # Wider figure
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.view_init(elev=0, azim=-90) 
    #ax1.set_xlim([-0.15, 2])  # Set X-axis limits from 0 to 1
    #ax1.set_ylim([-0.15, 1])  # Set Y-axis limits from 0 to 1
    #ax1.set_zlim([0.5, 3])  # Set Z-axis limits from 0 to 1
    # Plot points
    ax1.scatter(x, y, z, c='blue', marker='o', s=50)
    
    # Add labels to each point
    for i in range(len(points)):
        ax1.text(x[i], y[i], z[i] + 0.1 + float(np.random.random()/2), labels[i], fontsize=6, color='black', ha='center',)

    if len(targets) > 0:
        points = np.array([
            obj['bbox_center'] for obj in result
            if 'A wall on the side of a building' not in obj['description'] and obj['id'] in targets 
        ])
        labels = [f"{obj['id']}: {obj['description']}" for obj in result
                if 'A wall on the side of a building' not in obj['description'] and obj['id'] in targets
                ]  # Labels for each point
        
        # Extract X, Y, Z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Plot points
        ax1.scatter(x, y, z, c='green', marker='o', s=50)
        
        # Add labels to each point
        for i in range(len(points)):
            ax1.text(x[i], y[i], z[i]+ 0.1 +float(np.random.random()/2), labels[i], fontsize=6, color='black', ha='center',)

    if len(anchors) > 0:
        points = np.array([
            obj['bbox_center'] for obj in result
            if 'A wall on the side of a building' not in obj['description'] and obj['id'] in anchors 
        ])
        labels = [f"{obj['id']}: {obj['description']}" for obj in result
                if 'A wall on the side of a building' not in obj['description'] and obj['id'] in anchors 
                ]  # Labels for each point
        
        # Extract X, Y, Z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Plot points
        ax1.scatter(x, y, z, c='red', marker='o', s=50)
        
        # Add labels to each point
        for i in range(len(points)):
            ax1.text(x[i], y[i], z[i]+ 0.1 +float(np.random.random()/2), labels[i], fontsize=6, color='black', ha='center',)
        # Labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

    if  len(targets) > 0 and len(anchors) > 0:
        for rel in relations:
            x1, y1, z1 = objects_by_id[rel[0]]
            x2, y2, z2 = objects_by_id[rel[1]]
            # Draw edge between two points
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'k-', linewidth=2)
            
            # Calculate midpoint for text placement
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            mid_z = (z1+ z2) / 2
            
            # Add text above the edge
            ax1.text(mid_x, mid_y, mid_z + 0.15, rel[2], fontsize=6, color='red', ha='center')

    plt.savefig(os.path.join(SAVE_PATH, f"3d_{save_filename}"), dpi=300, bbox_inches='tight')
    
    information = f"User query: {user_query}\n"

    with open(os.path.join(SAVE_PATH, f"{save_filename.split('.')[0]}.txt"), "w") as f:
        f.write(information)    
    
    LLM_answer = str(LLM_answer).replace('\n', '').replace('\\', '')
    information = f"LLM answer: {LLM_answer}"

    with open(os.path.join(SAVE_PATH, f"{save_filename.split('.')[0]}.txt"), "a") as f:
        f.write(information)    

    for obj in result:
        if 'A wall on the side of a building' in obj['description'] or (int(obj['id']) not in targets and int(obj['id']) not in anchors):
            #print("Filtered 3D", int(obj['id']), targets, anchors)
            continue

        cx, cy, cz = np.array(obj['bbox_center'])
        dx, dy, dz = np.array(obj['bbox_extent']) / 2.0
        box_3d = np.array([
            [cx - dx, cy - dy, cz - dz],  # Front-bottom-left
            [cx + dx, cy - dy, cz - dz],  # Front-bottom-right
            [cx + dx, cy + dy, cz - dz],  # Front-top-right
            [cx - dx, cy + dy, cz - dz],  # Front-top-left
            [cx - dx, cy - dy, cz + dz],  # Back-bottom-left
            [cx + dx, cy - dy, cz + dz],  # Back-bottom-right
            [cx + dx, cy + dy, cz + dz],  # Back-top-right
            [cx - dx, cy + dy, cz + dz],  # Back-top-left
        ])

        box_2d = project_point_cloud_to_image(segmentation, depth, camera_pose, intrinsics, box_3d)
        #print(box_2d)
        
        # Define box edges (pairs of points to connect)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connect front and back
        ]
        
        # Draw the bounding box edges
        for (i, j) in edges:
            #print(tuple(box_2d[i]), tuple(box_2d[j]))
            cv2.line(segmentation, tuple(box_2d[i]), tuple(box_2d[j]), (0, 0, 255), 1)
        
        text = f"{obj['id']}: {obj['description']}"
        # Get the text size to create a background rectangle
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Define the rectangle's top-left and bottom-right corners (padding the text a bit)
        top_left = (box_2d[0][0] - 10, box_2d[0][1] + 10)
        bottom_right = (box_2d[0][0] + text_width + 10, box_2d[0][1] - text_height - 10)

        # Draw a white rectangle as the background
        cv2.rectangle(segmentation, top_left, bottom_right, (255, 255, 255), thickness=cv2.FILLED)

        cv2.putText(segmentation, text, (box_2d[0][0], box_2d[0][1]), font, font_scale, COLOR_CV, thickness, cv2.LINE_AA)

    for rel in relations:
        if rel[0] not in targets:
            continue
        x1, y1, z1 = objects_by_id[rel[0]]
        x2, y2, z2 = objects_by_id[rel[1]]

        line_3d = np.array([
            [x1, y1, z1],  # center 1
            [x2, y2, z2],  # bbox2
        ])

        line2d = project_point_cloud_to_image(segmentation, depth, camera_pose, intrinsics, line_3d)
        #print(line2d)
        cv2.line(segmentation, tuple(line2d[0]), tuple(line2d[1]), (255, 255, 0), 2)

        mid_x = int((line2d[0][0] + line2d[1][0]) / 2)
        mid_y = int((line2d[0][1] + line2d[1][1]) / 2)
        wrapped_text = textwrap.fill(rel[2], width=15)
        y0 = mid_y-25
        for i, line in enumerate(wrapped_text.split('\n')):

            # Get the text size to create a background rectangle
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            y = y0 + i*text_height
            # Define the rectangle's top-left and bottom-right corners (padding the text a bit)
            top_left = (mid_x-25, y)
            bottom_right = (mid_x-25 + text_width, y - text_height)

            # Draw a white rectangle as the background
            cv2.rectangle(segmentation, top_left, bottom_right, (255, 255, 255), thickness=cv2.FILLED)

            cv2.putText(segmentation, line, (mid_x-25, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(SAVE_PATH, f"overlayed_masks_sam_and_graph_{save_filename}"), segmentation)

def main():
    """Функция, выполняемая при получении сообщения от клиента."""
    print("Сообщение получено! Выполняем main()...")
    with open(TEXT_FILE, "r") as file:
        user_query = file.read().strip()
        print(f"Содержимое файла: {user_query}")


    if SAVE_PATH:
         os.makedirs(SAVE_PATH, exist_ok=True)
         with gzip.open(os.path.join(SAVE_PATH, "meta.pkl.gz"), "wb") as file:
            pickle.dump({"config": config}, file)
        
    logger.info(f"Parsed arguments. Utilizing config from {CONFIG_FILE}.")

    #rgbd_dataset = get_dataset(config["dataset"])
    # See Section 3.1
    logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
    color_path = COLOR_PATH
    #depth_path = self.depth_paths[index]
    color = np.asarray(imageio.imread(color_path), dtype=float)
    
    color = color[:, :, :3] # BGRA -> BGR
    #color = color[:, :, ::-1] # BGR -> RGB 

    color = torch.from_numpy(color.copy())

    color = color.to("cuda").type(torch.float)
    
    depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float64)
    depth = np.expand_dims(depth, -1)
    depth = torch.from_numpy(depth).to("cuda") / DEPTH_SCALE

    pose = np.loadtxt(POSE_PATH)
    #pose = np.array([
    #    [1.0, 0.0, 0.0, 0.0],
    #    [0.0, 1.0, 0.0, 0.0],
    #    [0.0, 0.0, 1.0, 0.0],
    #    [0.0, 0.0, 0.0, 1.0],
    #])
    #pose = torch.from_numpy(pose).to("cuda")

    # global FIRST_POSE
    # if FIRST_POSE is None:
    #     FIRST_POSE = torch.clone(pose)
    #     pose = torch.from_numpy(np.array([
    #         [1.0, 0.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]))
    # else:
    #     pose = relative_transformation(
    #                 FIRST_POSE,
    #                 pose,
    #                 orthogonal_rotations=False,
    #            )

    #print("FIRST POSE", pose, FIRST_POSE)

    pose = torch.from_numpy(np.array([
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0],
         ]))
    frame = (color, depth, INTRINSICS, pose)
    segmentation = nodes_constructor.integrate(0, frame,
        SAVE_PATH)

    cv2.imwrite(os.path.join(SAVE_PATH, "overlayed_masks_sam.png"), segmentation)

    torch.cuda.empty_cache()
    #nodes_constructor.postprocessing()
    torch.cuda.empty_cache()
    if SAVE_PATH:
        results = {'objects': nodes_constructor.objects.to_serializable()}
        with gzip.open(os.path.join(SAVE_PATH,
            f"frame_last_objects.pkl.gz"), "wb") as f:
                pickle.dump(results, f)

    # See Section 3.3
    logger.info('Captioning 3D objects.')
    objects = nodes_constructor.objects
    result = describe_objects(objects)
    torch.cuda.empty_cache()

    logger.info('Saving objects.')
    with open(os.path.join(SAVE_PATH, "objects.json"), "w") as f:
        json.dump(result, f, indent=2)

    #with open(os.path.join(SAVE_PATH, "objects.json"), "r") as f:
    #    result = json.load(f)
    
    
    if not DEBUG:
        llm.set_scene(os.path.join(SAVE_PATH, "objects.json"))
        logger.info("Selecting relevant nodes")
        related_objects, relations, json_answer = llm.select_relevant_nodes(user_query)
        logger.info(related_objects)

        targets = [obj['id'] for obj in related_objects['target_objects']]
        anchors = [obj['id'] for obj in related_objects['anchor_objects']]

    else:
        relations = [ (0, 1, "test relation"), (2, 3, "test relation")]
        
        json_answer = '{"test_answer": "Test_answer"}'


        targets = []
        anchors = []

        for obj in result:
            #if obj['id'] % 2 == 0:
            #    targets.append(obj['id'])
            #elif obj['id'] % 3 == 0:
            #    anchors.append(obj['id'])
            targets.append(3)
            anchors.append(0)


    segmentation1 = deepcopy(segmentation)

    draw_answer(result, targets, anchors, relations, segmentation1, depth, INTRINSICS, pose, user_query, json_answer, "relevant_objects.png")

    if DEBUG:
        return
    logger.info("Selecting reffered object")
    full_answer, final_answer, relations, pretty_answer = llm.select_referred_object(user_query, related_objects)
    logger.info(full_answer)

    targets = [obj['id'] for obj in related_objects['target_objects'] if obj['id'] == final_answer]
    filtered_relations = []

    for rel in relations:
        #print(rel, final_answer)
        if rel[0] == final_answer:
            targets.append(rel[0])
            anchors.append(rel[1])
            filtered_relations.append(rel)


    json_answer = pretty_answer
    segmentation2 = deepcopy(segmentation)

    draw_answer(result, targets, anchors, filtered_relations, segmentation2, depth, INTRINSICS, pose, user_query, json_answer, "final_answer.png")

def wait_for_message():
    """Ожидает изменения файла text.txt и вызывает main() при обновлении."""
    last_mod_time = None

    while True:
        if os.path.exists(TEXT_FILE):
            mod_time = os.path.getmtime(TEXT_FILE)
            if last_mod_time is None or mod_time > last_mod_time:
                last_mod_time = mod_time
                main()
        time.sleep(1)  # Проверка раз в секунду

if __name__ == "__main__":
    print("Ожидание сообщений от клиента...")
    wait_for_message()
