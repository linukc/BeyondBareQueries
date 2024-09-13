import os
import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from bbq.grounding.utils import get_semantic_edge


class Llama3:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.cuda()

        self.system_prompt_select_objects = """
            You are a helpful assistant.
            The user will describe a 3D scene using a list of objects
            placed in the scene. Each object is described by its
            semantic type and its object id. The user will ask you questions
            about this scene.
            """

        self.system_prompt_final_choice = """
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
            4. "description": a brief (but sometimes inaccurate)
            tag categorizing the object
            5. "relations": a list of strings (may be empty) describing spatial
            relations between this object and other objects in the scene. It contains 
            types of relations and pre-computed Euclidean distances between objects.
        """

    def set_scene(self, scene_description_path):
        with open(os.path.join(scene_description_path), "r") as f:
            self.scene_description = json.load(f)
        
        self.scene_center_point = np.mean([ob["bbox_center"] for ob in 
            self.scene_description], axis=0)

    def select_relevant_nodes(self, object_query):
        scene_desc = f"The list of objects in the scene is the following. "
        for ob in self.scene_description:
            scene_desc += f"the {ob['description'].split('.')[0].lower()} with id {ob['id']}, "

        user_query = f"{scene_desc}. The user's query: {object_query}. \
            Which objects are referred by the user based on their semantics? \
            If there are several objects of the same semantic type, choose all of them. \
            Which other objects you need to know the location to answer the user's query? \
            If the list of objects in the scene contains several objects of the same semantic type, choose all of them."
        user_query+= """
        Use the following format for the answer:
            {
                "referred objects": ["object1 with id 1", "object1 with id 2"] objects referred by the user based on their semantics,
                "anchors": ["object2 with id 4", "object3 with id 5", "object4 with id 6"], other objects which location you need to know to select the object referenced by the user's query. If the list of objects in the scene contains several objects of the same semantic type, choose all of them. 
            }
        """

        messages = [
            {"role": "system", "content": self.system_prompt_select_objects},
            {"role": "user", "content": user_query }
        ]
        
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.cuda()
        generated_ids = self.model.generate(model_inputs, 
            max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        LLMAnswer = decoded[0].split("assistant")[-1]

        json_answer = {
            "referred objects": [],
            "anchors": []
        }
        try:
            json_answer["referred objects"] = json.loads(
                LLMAnswer.split('"referred objects": ')[-1].split("]")[0] + "]")
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
            for ob in self.scene_description:
                if ob["id"] == index:
                    target_objects.append({
                        "id": ob["id"],
                        "bbox_center": [round(float(x),2) for x in ob["bbox_center"]],
                        "bbox_extent": [round(float(x),2) for x in ob["bbox_extent"]],
                        "description": ob["description"],

                    })
        for index in anchor_ids:
            for ob in self.scene_description:
                if ob["id"] == index and not "wall" in ob["description"]:
                    anchor_objects.append({
                        "id": ob["id"],
                        "bbox_center": [round(float(x),2) for x in ob["bbox_center"]],
                        "bbox_extent": [round(float(x),2) for x in ob["bbox_extent"]],
                        "description": ob["description"],

                    })

        related_objects = {
            "scene_desc": scene_desc, 
            "query": object_query, 
            "target_objects": target_objects,
            "anchor_objects": anchor_objects
        }
        return related_objects

    def select_referred_object(self, object_query, related_objects_dict):
        target_objects = related_objects_dict["target_objects"]
        anchor_objects = related_objects_dict["anchor_objects"]

        for i, ob1 in enumerate(target_objects):
            ob1['relations'] = []
            for j, ob2 in enumerate(anchor_objects):
                if ob1['id'] == ob2['id']:
                    continue
                
                rel_string = ""
                rels = get_semantic_edge(ob1["bbox_center"], ob2["bbox_center"], self.scene_center_point)
                distance = np.linalg.norm(np.array(ob1["bbox_center"]) - np.array(ob2["bbox_center"]))
                rels.append(f"at distance {np.round(distance,2)} m")
                if len(rels) > 0:
                    rel_string += f'The {ob1["description"]} with id {ob1["id"]} is {" and ".join(rels)} from the {ob2["description"]} with ids {ob2["id"]}.'
                if len(rel_string) > 2:
                    ob1['relations'].append(rel_string)

        related_objects = []
        for obj in target_objects:
            related_objects.append(obj)

        user_query = f"The JSON describing the relevant objects in the scene: {str(related_objects)},"
        user_query += f"Select objects that correspond the best to the query. \
            Deduce spatial relations between objects, using 'relations' field of JSON. The query: {object_query}."
        user_query += """
            Give me the id of selected object. Then explain me why you choose this object.
            Use the following format for the answer:
            {
                "explanation": your explanation,
                "id": id of the selected object
            }
        """

        messages = [
            {"role": "system", "content": self.system_prompt_final_choice},
            {"role": "user", "content": user_query }
        ]
        encodeds = self.tokenizer.apply_chat_template([messages[-1]], return_tensors="pt")
        model_inputs = encodeds.cuda()

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        LLMAnswer = decoded[0].split('"id": ')[-1]
        pred = int(''.join(c if c.isdigit() else '' for c in LLMAnswer.split("}")[0]))
        return decoded[0], pred
