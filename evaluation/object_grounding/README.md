This folder contains data and code for reproducing 3D object grounding experiments in our paper. It includes LLM inference code as well as evaluation scripts.

1. We provide the subset of Sr3D+/Nr3D, ScanRefer queries used in our experiments in the [data/](data) folder.

## Compare edge types on GT point clouds on Nr3D dataset.

Go to [gt_scripts/llm_inference](gt_scripts/llm_inference)

1. To select target and anchors objects for each query run:

```
python object_grounding_stage1_nr3d.py
```

Since LLM generation is probabilistic, it will ensure that we use the same set of targets and anchors for different type of edges.

The following scripts will run LLAMA3-8B-Instruct and create a folder with saved LLM output in .txt files.

2. To run 3D object grounding experiment without edges between target and anchor objects in the scene graph run:

```
python object_grounding_stage2_nr3d_no_edges.py
```

3. To run 3D object grounding experiment with semantic edges between target and anchor objects in the scene graph run:

```
python object_grounding_stage2_nr3d_semantic.py
```

4. To run 3D object grounding experiment with metric edges between target and anchor objects in the scene graph run:

```
python object_grounding_stage2_nr3d_metric.py
```

5. To run 3D object grounding experiment with semantic and metric edges between target and anchor objects in the scene graph run:

```
python object_grounding_stage2_nr3d_semantic_and_metric.py
```

6. To compute Recall@1 metric go to [gt_scripts/evaluation](gt_scripts/evaluation) and run:

```
python recall_nr3d.py
```

Adjust paths in the beginning of each scripts to reflect your folder structures.

## Run main experiments.

We use GPT4-o and BatchAPI in our experiments. Add your OpenAI token to your environment variable ```OPENAI_API_KEY```. Each folder consists of 4 scripts to:

- prepare batch (prepare_batch_*.py)
- send batch through BatchAPI (send_batchapi.py)
- check batch completion (check_batch.py)
- retrieva batch once completed (retrieve_batch.py)


### on Sr3D+ dataset.

1. To select target and anchors objects for each query go to [pred_scripts/llm_inference/sr3d/select_relevant_objects_bbq](pred_scripts/llm_inference/sr3d/select_relevant_objects_bbq) and run:

```
python prepare_batch_input_sr3d.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

2.  To run 3D object grounding experiment with semantic and metric edges between target and anchor objects in the scene graph go to [pred_scripts/llm_inference/sr3d/object_grounding_semantic_and_metric_edges_bbq/](pred_scripts/llm_inference/sr3d/object_grounding_semantic_and_metric_edges_bbq/) and run:

```
python prepare_batch_input_sr3d.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

3. To compute evaluation metrics go to [pred_scripts/evaluation/](pred_scripts/evaluation/)run:

```
python compute_accuracy_bbq_objects_sr3d.py
```

### on Nr3D dataset.

1. To select target and anchors objects for each query go to [pred_scripts/llm_inference/nr3d/select_relevant_objects_bbq](pred_scripts/llm_inference/nr3d/select_relevant_objects_bbq) and run:

```
python prepare_batch_input_nr3d.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

2.  To run 3D object grounding experiment with semantic and metric edges between target and anchor objects in the scene graph go to [pred_scripts/llm_inference/nr3d/object_grounding_semantic_and_metric_edges_bbq/](pred_scripts/llm_inference/nr3d/object_grounding_semantic_and_metric_edges_bbq/) and run:

```
python prepare_batch_input_nr3d.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

3. To compute evaluation metrics go to [pred_scripts/evaluation/](pred_scripts/evaluation/)run:

```
python compute_accuracy_bbq_objects_nr3d.py
```

### on ScanRefer dataset.

1. To select target and anchors objects for each query go to [pred_scripts/llm_inference/scanrefer/select_relevant_objects_bbq](pred_scripts/llm_inference/scanrefer/select_relevant_objects_bbq) and run:

```
python prepare_batch_input_scanrefer.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

2.  To run 3D object grounding experiment with semantic and metric edges between target and anchor objects in the scene graph go to [pred_scripts/llm_inference/scanrefer/object_grounding_semantic_and_metric_edges_bbq/](pred_scripts/llm_inference/scanrefer/object_grounding_semantic_and_metric_edges_bbq/) and run:

```
python prepare_batch_input_scanrefer.py
python send_batchapi.py
python check_batch.py
python retrieve_batch.py
```

3. To compute evaluation metrics go to [pred_scripts/evaluation/](pred_scripts/evaluation/)run:

```
python compute_accuracy_bbq_objects_scanrefer.py
```
