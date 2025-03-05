import os
from collections import defaultdict
import json
import csv
import numpy as np

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)
    #print(inter_vol, "pred:", box_vol_1, "gt:", box_vol_2)
    return iou

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max

keys = [
    "mentions_target_class",
    "uses_object_lang",
    "uses_spatial_lang",
    "uses_color_lang",
    "uses_shape_lang",
    "is_easy",
    "is_view_dep"
]

tp = {
    k: {
        "True": {
            "0.1": 0,
            "0.25": 0
        },
        "False": {
            "0.1": 0,
            "0.25": 0
        },
    }

    for k in keys
}

tp["overall"] = {
    "0.1": 0,
    "0.25": 0
}

total_number = {
    k: {
        "True": 0,
        "False": 0,
    }
    for k in keys
}

total_number["overall"] = 0


gpt_answer = "../llm_inference/nr3d/object_grounding_semantic_and_metric_edges_bbq/gpt_response_related_objects_nr3d.jsonl"
gt_directory = "../../data/scene_description_unaligned_gt/"
annotation_directory = "../../data/scannet/nr3d_all_types_of_queries"
inst_sem_directory = "../../data/bbq/"

with open(gpt_answer, "r") as f:
    retrieved_objects = json.load(f)

retrieved_objects_dict = {
    output["custom_id"]: json.loads(
        output["response"]["body"]["choices"][0]["message"]["content"]
    )
    for output in retrieved_objects
}

scenes = [
    "scene0011_00",
    "scene0030_00",
    "scene0046_00",
    "scene0086_00",
    "scene0222_00",
    "scene0378_00",
    "scene0389_00",
    "scene0435_00"
]

for scene in scenes:

    gt_scene_file = os.path.join(gt_directory, f"{scene}_scene_description_unaligned.json")
    annotation_file = os.path.join(annotation_directory, f"{scene}_annotation.json")
    inst_segm_scene_file = os.path.join(
        inst_sem_directory,
        f"09.06.2024_scannet_{scene}.json"
    )

    with open(gt_scene_file, "r") as f:
        gt_objects = json.load(f)

    with open(inst_segm_scene_file, "r") as f:
        inst_segm_objects = json.load(f)

    with open(annotation_file, "r") as f:
        anns = json.load(f)

    for i, ann in enumerate(anns):
        obj_id = ann["target_id"]
        request_custom_id = f"{scene}_{ann['ann_id']}_{ann['target_id']}"
        pred_id = retrieved_objects_dict[request_custom_id]["id"]
        for ob in gt_objects:
            if ob['id'] == obj_id:
                gt_corners = construct_bbox_corners(ob['bbox_center'], ob['bbox_extent'])
        
        for ob in inst_segm_objects:
            if ob['id'] == pred_id:
                pred_corners = construct_bbox_corners(ob['bbox_center'], ob['bbox_extent'])
                pred_center = ob['bbox_center']
        iou = box3d_iou(pred_corners, gt_corners)

        for k in keys:
            if scene == "scene0435_00" and k == "is_view_dep" and str(anns[i][k]) == "True" and iou >= 0.5:
                print(ann['utterance'])
                print(request_custom_id)
                print("GT:",  obj_id)
                print("pred_id", pred_id)
            if iou >= 0.25:
                tp[k][str(anns[i][k])]["0.25"] += 1
            if iou >= 0.1:
                tp[k][str(anns[i][k])]["0.1"] += 1
            total_number[k][str(anns[i][k])] += 1
        
        if iou >= 0.25:
            tp["overall"]["0.25"] += 1
        if iou >= 0.1:
            tp["overall"]["0.1"] += 1
        total_number["overall"] += 1


for k in keys:
    for v in ["False", "True"]:
        for iou in ["0.1", "0.25"]:
            print(k, v, iou, np.round(100* tp[k][v][iou]/ total_number[k][v], 1))

for iou in ["0.1", "0.25"]:
    print("Overall", iou, np.round(100* tp["overall"][iou]/ total_number["overall"],1))