import requests
import os
import json
import numpy as np
from ast import literal_eval
import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
#from unsloth import FastLanguageModel
import torch
import pandas as pd
from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras

RANDOM_SEED = 2024
scene_descr_dir = "../../data/scene_description_unaligned_gt"
output_dir = "output/gt_nr3d_semantic_and_metric_edges"
os.makedirs(output_dir, exist_ok=True)
annotation_dir = f"../../data/scannet/nr3d_all_types_of_queries"

model = AutoModelForCausalLM.from_pretrained("../../Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("../../Meta-Llama-3-8B-Instruct", padding_side='left')
tokenizer.padding_side = "left"
model.cuda()
model.eval()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
out_of_memory_queries = []



related_object_dir = f"output/gt_related_objects_nr3d"

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

system_prompt_final_choice = """
            You are a helpful assistant.
            The user will describe a 3D scene described in a
            JSON format. The JSON describes
            objects with the following
            five fields:
            1. "id": a unique object id
            2. "bbox_center": centroid of the 3D bounding box
            for the object
            3. "bbox_extent": extents of the 3D bounding box
            for the object
            4. "object_tag": a brief (but sometimes inaccurate)
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
    with open(os.path.join(scene_descr_dir, f"{scene}_scene_description_unaligned.json"), "r") as f:
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

        ann_target = ann["target_id"]
        output_file_name = f"{scene}_{ann_id}_{ann['target_id']}.txt"
        if os.path.exists(os.path.join(output_dir, output_file_name)):
            print("skipping ", os.path.join(output_dir, output_file_name))
            continue
        related_objects_json = f"{scene}_{ann_id}_{ann['target_id']}.json"
        
        with open(os.path.join(related_object_dir, related_objects_json), "r") as f:
            related_objects_dict = json.load(f)

        target_objects = related_objects_dict["target_objects"]
        anchor_objects = related_objects_dict["anchor_objects"]

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
                    rel_string += f'The {ob1["object_tag"]} with id {ob1["id"]} is {" and ".join(rels)} from the {ob2["object_tag"]} with ids {ob2["id"]}.'
                if len(rel_string) > 2:
                    ob1['relations'].append(rel_string)

        related_objects = []
        for obj in target_objects:
            related_objects.append(obj)

        user_query = f"The JSON describing the relevant objects in the scene: {str(related_objects)},"

        user_query += f"Select objects that correspond the best to the query. Deduce spatial relations between objects, using 'relations' field of JSON. The query: {ann_utterance}."


        user_query += """
            Give me the id of selected object. Then explain me why you choose this object.
            Use the following format for the answer:
            {
                "explanation": your explanation,
                "id": id of the selected object
            }
        """

        messages = [
            {"role": "system", "content": system_prompt_final_choice},
            {"role": "user", "content": user_query }
        ]

        encodeds = tokenizer.apply_chat_template([messages[-1]], return_tensors="pt")
        model_inputs = encodeds.cuda()
        try:
            with torch.no_grad():
                generated_ids = model.generate(model_inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, retrying batch')
                out_of_memory_queries.append(f"{scene}_{ann_id}_{ann['target_id']}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        decoded = tokenizer.batch_decode(generated_ids)
        with open(os.path.join(output_dir, output_file_name), "a") as f:
            f.write(decoded[0])
        print(decoded[0])
        print("GT target:", ann["target_id"])
        try:
            LLMAnswer = decoded[0].split('"id": ')[-1]
            pred = int(''.join(c if c.isdigit() else '' for c in LLMAnswer.split("}")[0]))
        except:
            messages.append({
                "role": "assistant",
                "content": decoded[0].split("assistant")[-1]
            })
            user_query = """
                You used wrong formatting for the answer. Please,
                format your previous answer using the following JSON format:
                {
                    "explanation": your explanation,
                    "id": id of the selected object
                }
            """
            messages.append({
                "role": "user",
                "content": user_query
            })
            encodeds = tokenizer.apply_chat_template([messages[-1]], return_tensors="pt")
            model_inputs = encodeds.cuda()
        try:
            with torch.no_grad():
                generated_ids = model.generate(model_inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, retrying batch')
                out_of_memory_queries.append(f"{scene}_{ann_id}_{ann['target_id']}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
            decoded = tokenizer.batch_decode(generated_ids)
            print(decoded[0])
            with open(os.path.join(output_dir, output_file_name), "a") as f:
                f.write(decoded[0])
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, "oom_queries.json"), "w") as f:
            json.dump(out_of_memory_queries, f)
