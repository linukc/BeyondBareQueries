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

RANDOM_SEED = 2024
scene_descr_dir = "../../data/scene_description_unaligned_gt"
output_dir = "output/gt_related_objects_nr3d"
os.makedirs(output_dir, exist_ok=True)
annotation_dir = f"../../data/scannet/nr3d_all_types_of_queries"


model = AutoModelForCausalLM.from_pretrained("../../Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("../../Meta-Llama-3-8B-Instruct", padding_side='left')

model.cuda()
model.eval()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
out_of_memory_queries = []

tokenizer.padding_side = "left"

system_prompt_select_objects = """
            You are a helpful assistant.
            The user will describe a 3D scene using a list of objects
            placed in the scene. Each object is described by its
            semantic type and its object id. The user will ask you questions
            about this scene.
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

    scene_desc = f"The list of objects in the scene is the following. "
    for ob in query:
        scene_desc += f"the {ob['object_tag'].split('.')[0]} with id {ob['id']}, "

    annotation_file = os.path.join(annotation_dir, f"{scene}_annotation.json")
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for ann_id, ann in enumerate(annotations):
        ann_utterance = ann["utterance"]

        ann_target = ann["target_id"]
        output_file_name = f"{scene}_{ann_id}_{ann['target_id']}.json"


        user_query = f"{scene_desc}. The user's query: {ann['utterance']}. Which objects are referred by the user based on their semantics? If there are several objects of the same semantic type, choose all of them. Which other objects you need to know the location to answer the user's query? If the list of objects in the scene contains several objects of the same semantic type, choose all of them."
        user_query+= """
        Use the following format for the answer:
            {
                "referred objects": ["object1 with id 1", "object1 with id 2"] objects referred by the user based on their semantics,
                "anchors": ["object2 with id 4", "object3 with id 5", "object4 with id 6"], other objects which location you need to know to select the object referenced by the user's query. If the list of objects in the scene contains several objects of the same semantic type, choose all of them. 
            }
        """

        messages = [
            {"role": "system", "content": system_prompt_select_objects},
            {"role": "user", "content": user_query }
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.cuda()

        try:
            with torch.no_grad():
                generated_ids = model.generate(model_inputs, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        except RuntimeError as e:
            if 'out of memory' in str(e) and not raise_oom:
                print('| WARNING: ran out of memory, retrying batch')
                out_of_memory_queries.append(f"{scene}_{ann_id}_{ann['target_id']}")
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained("../../Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained("../../Meta-Llama-3-8B-Instruct")

                model.cuda()
                model.eval()
                continue
            else:
                raise e


        decoded = tokenizer.batch_decode(generated_ids)
        LLMAnswer = decoded[0].split("assistant")[-1]

        messages.append({
            "role": "assistant",
            "content": LLMAnswer
        })

        json_answer = {
            "referred objects": [],
            "anchors": []
        }
        try:
            json_answer["referred objects"] = json.loads(LLMAnswer.split('"referred objects": ')[-1].split("]")[0] + "]")
        except:
            pass

        try:
            json_answer["anchors"]= json.loads(LLMAnswer.split('"anchors": ')[-1].split("]")[0] + "]")
        except:
            pass

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
                        "object_tag": ob['object_tag'],

                    })
        for index in anchor_ids:
            for ob in query:
                if ob["id"] == index:
                    anchor_objects.append({
                        "id": ob["id"],
                        "bbox_center": [round(float(x),2) for x in ob["bbox_center"]],
                        "bbox_extent": [round(float(x),2) for x in ob["bbox_extent"]],
                        "object_tag": ob['object_tag'],

                    })

        related_objects = {
            "scene_desc": scene_desc, 
            "query": ann['utterance'], 
            "target_objects": target_objects,
            "anchor_objects": anchor_objects
        }
        with open(os.path.join(output_dir, output_file_name), "w") as f:
            json.dump(related_objects, f)
        print(decoded[0])
        torch.cuda.empty_cache()
        with open(os.path.join(output_dir, "oom_queries.json"), "w") as f:
            json.dump(out_of_memory_queries, f)
