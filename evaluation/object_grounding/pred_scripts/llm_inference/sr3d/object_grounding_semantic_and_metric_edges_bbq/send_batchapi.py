from openai import OpenAI
import os
import json

client = OpenAI(
   api_key = os.environ["OPENAI_API_KEY"]
)

batch_input_file = client.files.create(
  file=open("batch_requests.jsonl", "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id

batch_object = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    },
)

print(batch_object)

# Convert batch_object to a dictionary
batch_dict = batch_object.model_dump()

# Save the dictionary as JSON
with open("example_batch_object.json", "w") as f:
    json.dump(batch_dict, f, indent=2)