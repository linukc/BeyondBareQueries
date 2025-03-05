import os
import json
import re
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras

RANDOM_SEED = 2024
scene_descr_dir = "../../../../data/bbq"
annotation_dir = f"../../../../data/scannet/ScanRefer_sample"

related_objects_json = "../select_relevant_objects_bbq/gpt_response_related_objects_scanrefer.jsonl"

with open(related_objects_json, "r") as f:
    related_objects = json.load(f)

related_objects_dict = {
    output["custom_id"]: json.loads(
        output["response"]["body"]["choices"][0]["message"]["content"]
    )
    for output in related_objects
}

request_list = []

def egoview_project(target, anchor, center):
    anchor_obj_loc = torch.tensor(anchor['obj_loc']).unsqueeze(0).float()
    cam_pos = torch.tensor(center).unsqueeze(0).float()
    R, T = look_at_view_transform(eye=cam_pos, at=anchor_obj_loc, up=((0.0, 0.0, 1.0),))
    camera = FoVOrthographicCameras(device='cpu', R=R, T=T)

    pos = torch.tensor(target['obj_loc']).unsqueeze(0).float()
    target_pos_2d = camera.transform_points_screen(pos, image_size=(512, 2048))
    anchor_pose_2d = camera.transform_points_screen(anchor_obj_loc, image_size=(512, 2048))

    return target_pos_2d, anchor_pose_2d

def get_semantic_edge(target, anchor, center_point):
    t = {"obj_loc": target}
    a = {"obj_loc": anchor}
    target_pos_2d, anchor_pose_2d = egoview_project(t, a, center_point)
    #print(target_pos_2d)
    #print(anchor_pose_2d)

    relations = []
    if target_pos_2d[0, 0] < anchor_pose_2d[0, 0]:
        relations.append("left")
    if target_pos_2d[0, 0] > anchor_pose_2d[0, 0]:
        relations.append("right")

    if target_pos_2d[0, 2] < anchor_pose_2d[0, 2]:
        relations.append("front")
    if target_pos_2d[0, 2] > anchor_pose_2d[0, 2]:
        relations.append("back")

    if target_pos_2d[0, 1] < anchor_pose_2d[0, 1]:
        relations.append("above")
    if target_pos_2d[0, 1] > anchor_pose_2d[0, 1]:
        relations.append("below")

    return relations


SystemPrompt = """
        You are an helpful assistant. 
        The user will describe a 3D scene described in a
        JSON format. The JSON describes objects with the following five fields:
        1. "id": a unique object id
        2. "bbox_center": centroid of the 3D bounding box
        for the object
        3. "bbox_extent": extents of the 3D bounding box
        for the object
        4. "description": a brief (but sometimes inaccurate)
        tag categorizing the object
        5. "relations": a list of strings (may be empty) describing spatial
        relations between this object and other objects in the scene. It contains 
        types of relations and pre-computed Euclidean distances between objects.
"""

scenes = [
    "scene0435_00",
    "scene0046_00",
    "scene0378_00",
    "scene0222_00",
    "scene0389_00",
    "scene0086_00",
    "scene0030_00",
    "scene0011_00",
]

for scene in scenes:
    with open(os.path.join(scene_descr_dir, f"09.06.2024_scannet_{scene}.json"), "r") as f:
        query = json.load(f)

    center_point = np.mean([
        ob["bbox_center"]
        for ob in query
    ], axis=0)

    annotation_file = os.path.join(annotation_dir, f"{scene}_annotation.json")
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for ann_id, ann in enumerate(annotations):

        ann_utterance = ann["utterance"]
        ann_target = ann["target"]

        request_custom_id = f"{scene}_{ann_id}_{ann['target']}"

        json_answer = related_objects_dict[request_custom_id]
        target_ids = []
        anchor_ids = []
        
        if "referred objects" in json_answer:
            for s in json_answer["referred objects"]:
                try:
                    target_ids.append(int(re.findall(r'\b\d+\b', s)[0]))
                except:
                    continue
        if "anchors" in json_answer:
            for s in json_answer["anchors"]:
                try:
                    anchor_ids.append(int(re.findall(r'\b\d+\b', s)[0]))
                except:
                    continue

        target_ids = set(target_ids)
        anchor_ids = set(anchor_ids)

        target_objects = []
        anchor_objects = []

        for index in target_ids:
            for ob in query:
                if ob["id"] == index:
                    target_objects.append({
                        "id": ob["id"],
                        "bbox_center": [round(float(x),2) for x in ob["bbox_center"]],
                        "bbox_extent": [round(float(x),2) for x in ob["bbox_extent"]],
                        "description": ob["description"],

                    })
        for index in anchor_ids:
            for ob in query:
                if ob["id"] == index:
                    anchor_objects.append({
                        "id": ob["id"],
                        "bbox_center": [round(float(x),2) for x in ob["bbox_center"]],
                        "bbox_extent": [round(float(x),2) for x in ob["bbox_extent"]],
                        "description": ob["description"],

                    })

        for i, ob1 in enumerate(target_objects):
            ob1['relations'] = []
            for j, ob2 in enumerate(anchor_objects):
                if ob1['id'] == ob2['id']:
                    continue

                rels = get_semantic_edge(ob1["bbox_center"], ob2["bbox_center"], center_point)
                rel_string = ""
                distance = np.linalg.norm(np.array(ob1["bbox_center"])-np.array(ob2["bbox_center"]))
                #rel_string = f'The {ob1["object_tag"]} with id {ob1["id"]} is at distance {np.round(distance,2)} m from the {ob2["object_tag"]} with ids {ob2["id"]}. '
                rels.append(f"at distance {np.round(distance,2)} m")
                if len(rels) > 0:
                    rel_string += f'The {ob1["description"].split(".")[0].lower()} with id {ob1["id"]} is {" and ".join(rels)} from the {ob2["description"].split(".")[0].lower()} with id {ob2["id"]}.'
                if len(rel_string) > 2:
                    ob1['relations'].append(rel_string)
        related_objects = []
        for obj in target_objects:
            related_objects.append(obj)

        user_query = f"The JSON describing the relevant objects in the scene: {str(related_objects)},"

        user_query += f"Select objects that correspond the best to the query. Deduce spatial relations between objects, using 'relations' field of JSON. The query: {ann_utterance}."


        user_query += """
            Give me the id of selected object. Then explain me why you choose this object.
            Use the following JSON format for the answer:
            {
                "explanation": your explanation,
                "id": id of the selected object
            }
        """


        request_dict = {
            "custom_id": request_custom_id, 
            "method": "POST",
            "url": "/v1/chat/completions",
            "body":
            {
            "model": "gpt-4o-2024-08-06",
            #"model": "gpt-4o-mini",
            "seed": 42,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": SystemPrompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
                ],
                "max_tokens": 1000
                }
        }
        request_list.append(request_dict)

with open('batch_requests.jsonl', 'w') as f:
    for item in request_list:
        json.dump(item, f)
        f.write('\n')