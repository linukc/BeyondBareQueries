import os
import json
from openai import OpenAI

client = OpenAI(
   api_key = os.environ["OPENAI_API_KEY"]
)

with open("example_batch_object.json", "r") as f:
    batch_data = json.load(f)

batch_object = client.batches.retrieve(batch_data["id"])

print(batch_object)
# Convert batch_object to a dictionary
batch_dict = batch_object.model_dump()

# Save the dictionary as JSON
with open("example_batch_object_retrieved.json", "w") as f:
    json.dump(batch_dict, f, indent=2)