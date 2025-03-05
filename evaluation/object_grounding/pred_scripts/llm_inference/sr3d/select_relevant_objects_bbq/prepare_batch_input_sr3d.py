import os
import json

RANDOM_SEED = 2024
scene_descr_dir = "../../../../data/bbq"
RELATIONS = [
    "above", "below", "supported", "supporting",
    "left", "right", "back", "front",
    "between", "farthest", "closest"
]

request_list = []

for RELATION in RELATIONS:
    annotation_dir = f"../../../../data/scannet/sr3d+/sr3d+_{RELATION}"

    SystemPrompt = """
            You are an helpful assistant. 
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
        with open(os.path.join(scene_descr_dir, f"09.06.2024_scannet_{scene}.json"), "r") as f:
            query = json.load(f)

        scene_desc = f"The list of objects in the scene is the following. "
        for ob in query:
            scene_desc += f"the {ob['description'].split('.')[0].lower()} with id {ob['id']}, "

        annotation_file = os.path.join(annotation_dir, f"{scene}_annotation.json")
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        for ann_id, ann in enumerate(annotations):

            ann_utterance = ann["utterance"]
            ann_target = ann["target_id"]

            request_custom_id = f"{scene}_{RELATION}_{ann_id}_{ann['target_id']}"

            user_query = f"{scene_desc}. The user's query: {ann['utterance']}. Which objects are referred by the user based on their semantics? If there are several objects of the same semantic type, choose all of them. Which other objects you need to know the location to answer the user's query? If the list of objects in the scene contains several objects of the same semantic type, choose all of them."
            user_query+= """
            Use the following JSON format for the answer:
                {
                    "referred objects": ["object1 with id 1", "object1 with id 2"] objects referred by the user based on their semantics,
                    "anchors": ["object2 with id 4", "object3 with id 5", "object4 with id 6"], other objects which location you need to know to select the object referenced by the user's query. If the list of objects in the scene contains several objects of the same semantic type, choose all of them. 
                }
            """

            request_dict = {
                "custom_id": request_custom_id, 
                "method": "POST",
                "url": "/v1/chat/completions",
                "body":
                {
                "model": "gpt-4o-2024-08-06",
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