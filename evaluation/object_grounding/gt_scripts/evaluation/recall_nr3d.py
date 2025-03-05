import os
import json
import shutil
import re

annotation_dir = f"../../data/scannet/nr3d_all_types_of_queries"
input_directory = "../llm_inference/output/gt_nr3d_semantic_edges/"

tp_0 = 0
tp_is_easy_true = 0
tp_is_easy_false = 0
tp_is_view_dep_true = 0
tp_is_view_dep_false = 0

number_of_failed_parsing = 0
total_number = 0
total_number_is_easy_true = 0
total_number_is_easy_false = 0
total_number_is_view_dep_true = 0
total_number_is_view_dep_false = 0

pred = {}

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
    annotation_file = os.path.join(annotation_dir, f"{scene}_annotation.json")
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for ann_id, ann in enumerate(annotations):

        ann_utterance = ann["utterance"]
        ann_target = ann["target_id"]

        request_custom_id = f"{scene}_{ann['ann_id']}_{ann['target_id']}"
        if not ann["uses_spatial_lang"] and not ann["mentions_target_class"]:
            continue


        total_number += 1
        if ann["is_easy"]:
            total_number_is_easy_true += 1
        else:
            total_number_is_easy_false += 1

        if ann["is_view_dep"]:
            total_number_is_view_dep_true += 1
        else:
            total_number_is_view_dep_false += 1


        try:
            with open(os.path.join(input_directory, f"{request_custom_id}.txt"), "r") as f:
                file_string = f.read()
            LLMAnswer = file_string.split('"id": ')[-1]
            pred_id = int(''.join(c if c.isdigit() else '' for c in LLMAnswer.split("}")[0]))
        except:
            print(f"{request_custom_id}.txt")
            try:
                print(LLMAnswer.split("}")[0])
            except:
                pass
            number_of_failed_parsing += 1
            continue

        if pred_id == ann_target:
            tp_0 += 1
            if ann["is_easy"]:
                tp_is_easy_true += 1
            else:
                tp_is_easy_false += 1

            if ann["is_view_dep"]:
                tp_is_view_dep_true += 1
            else:
                tp_is_view_dep_false += 1


print("Total number of queries:", total_number)
print("Recall at 1", tp_0/total_number)

print("Total number of queries easy:", total_number_is_easy_true)
print("Recall at 1", tp_is_easy_true/total_number_is_easy_true)

print("Total number of queries hard:", total_number_is_easy_false)
print("Recall at 1", tp_is_easy_false/total_number_is_easy_false)

print("Total number of queries view dependent:", total_number_is_view_dep_true)
print("Recall at 1", tp_is_view_dep_true/total_number_is_view_dep_true)

print("Total number of queries view independent:", total_number_is_view_dep_false)
print("Recall at 1", tp_is_view_dep_false/total_number_is_view_dep_false)

print("Invalid answers rate:", number_of_failed_parsing/total_number)