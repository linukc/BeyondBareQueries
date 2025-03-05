import os
import json
from openai import OpenAI

client = OpenAI(
   api_key = os.environ["OPENAI_API_KEY"]
)

with open("example_batch_object_retrieved.json", "r") as f:
    batch_data = json.load(f)

file_response = client.files.content(batch_data["output_file_id"])

json_outputs = []
for line in file_response.text.splitlines():
   json_output = json.loads(line)
   json_outputs.append(json_output)

with open("gpt_response_related_objects_nr3d.jsonl", "w") as f:
    json.dump(json_outputs, f, indent=2)