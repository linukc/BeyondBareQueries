import os
from collections import defaultdict
import json
import csv
import numpy as np



    
def scanrefer_get_unique_multiple_lookup():
    unique_multiple_lookup_file = '../../data/scanrefer_unique_multiple_lookup.json'
    if not os.path.exists(unique_multiple_lookup_file):
        type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
        scannet_labels = type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        label_classes_set = set(scannet_labels)
        raw2label = {}
        with open('../../data/scannet/scannetv2-labels.combined.tsv', 'r') as f:
            csvreader = csv.reader(f, delimiter='\t')
            csvreader.__next__()
            for line in csvreader:
                raw_name = line[1]
                nyu40_name = line[7]
                if nyu40_name not in label_classes_set:
                    raw2label[raw_name] = scannet2label['others']
                else:
                    raw2label[raw_name] = scannet2label[nyu40_name]
        all_sem_labels = defaultdict(list)
        cache = defaultdict(dict)
        scanrefer_data = json.load(open('../../data/scanrefer/ScanRefer_filtered_val.json'))
        for data in scanrefer_data:
            scene_id = data['scene_id']
            object_id = data['object_id']
            object_name = ' '.join(data['object_name'].split('_'))
            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(raw2label[object_name])
                except:
                    all_sem_labels[scene_id].append(17)
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}
        unique_multiple_lookup = defaultdict(dict)
        for data in scanrefer_data:
            scene_id = data['scene_id']
            object_id = data['object_id']
            object_name = ' '.join(data['object_name'].split('_'))
            try:
                sem_label = raw2label[object_name]
            except:
                sem_label = 17
            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
            unique_multiple_lookup[scene_id][object_id] = unique_multiple
        with open(unique_multiple_lookup_file, 'w') as f:
            json.dump(unique_multiple_lookup, f, indent=4)
    else:
        unique_multiple_lookup = json.load(open(unique_multiple_lookup_file))
    return unique_multiple_lookup

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


unique_multiple_lookup = scanrefer_get_unique_multiple_lookup()

iou25_acc = 0
iou10_acc = 0
unique_iou25_acc = 0
unique_iou10_acc = 0
unique_all = 0
multiple_iou25_acc = 0
multiple_iou10_acc = 0
multiple_all = 0

gpt_answer = "../llm_inference/scanrefer/object_grounding_semantic_and_metric_edges_bbq/gpt_response_related_objects_scanrefer.jsonl"
with open(gpt_answer, "r") as f:
    retrieved_objects = json.load(f)

retrieved_objects_dict = {
    output["custom_id"]: json.loads(
        output["response"]["body"]["choices"][0]["message"]["content"]
    )
    for output in retrieved_objects
}
gt_directory = "../../data/scene_description_unaligned_gt/"
annotation_directory = "../../data/scannet/ScanRefer_sample"
inst_sem_directory = "../../data/bbq/"

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

total_number = 0
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
        total_number += 1

        obj_id = anns[i]["target"]
        request_custom_id = f"{scene}_{i}_{ann['target']}"
        pred_id = retrieved_objects_dict[request_custom_id]["id"]

        for ob in gt_objects:
            if ob['id'] == obj_id:
                gt_corners = construct_bbox_corners(ob['bbox_center'], ob['bbox_extent'])
        
        for ob in inst_segm_objects:
            if ob['id'] == pred_id:
                pred_corners = construct_bbox_corners(ob['bbox_center'], ob['bbox_extent'])
                pred_center = ob['bbox_center']
        
        unique_multiple = unique_multiple_lookup[scene][str(obj_id)]
        if unique_multiple == 0:
            unique_all += 1
        else:
            multiple_all += 1
 
        iou = box3d_iou(pred_corners, gt_corners)
        if iou >= 0.25:
            iou25_acc += 1
            if unique_multiple == 0:
                unique_iou25_acc += 1
            else:
                multiple_iou25_acc += 1
            # iou25_acc_list[scannet_locs.shape[0]] += 1
        if iou >= 0.1:
            iou10_acc += 1
            if unique_multiple == 0:
                unique_iou10_acc += 1
            else:
                multiple_iou10_acc += 1
            # iou10_acc_list[scannet_locs.shape[0]] += 1
        # count_list[scannet_locs.shape[0]] += 1

val_scores = {
    '[scanrefer] Acc@0.25': float(iou25_acc) / total_number,
    '[scanrefer] Acc@0.1': float(iou10_acc) / total_number,
    '[scanrefer] Unique Acc@0.25': float(unique_iou25_acc) / unique_all,
    '[scanrefer] Unique Acc@0.1': float(unique_iou10_acc) / unique_all,
    '[scanrefer] Multiple Acc@0.25': float(multiple_iou25_acc) / multiple_all,
    '[scanrefer] Multiple Acc@0.1': float(multiple_iou10_acc) / multiple_all
}

for k, v in val_scores.items():
    print(k, np.round(100*v, 1))