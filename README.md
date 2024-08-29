<p align="center">

  <h1 align="center">Beyond Bare Queries: Open-Vocabulary Object Retrieval with 3D Scene Graph</h1>
  <p align="center">
    <a href="https://github.com/linukc"><strong>Linok Sergey</strong></a>
    ·
    <a href="https://github.com/wingrune"><strong>Tatiana Zemskova</strong></a>
    ·
    <a href=""><strong>Svetlana Ladanova</strong></a>
    ·
    <a href=""><strong>Roman Titkov</strong></a>
    ·
    <a href=""><strong>Dmitry Yudin</strong></a>
  </p>

  <h4 align="center"><a href="https://arxiv.org/abs/2406.07113">Paper</a> | <a href="https://arxiv.org/abs/2406.07113">arXiv</a> | <a href="https://linukc.github.io/BeyondBareQueries/">Project Page</a></h4>
  <div align="center"></div>
</p>

<p align="center">
<img src="assets/pipeline.png" width="80%">
</p>

## Getting Started

### System Requirements
18GB+ vRAM

### Data Preparation

#### Replica
Download the [Replica](https://github.com/facebookresearch/Replica-Dataset) RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in [Nice-SLAM](https://github.com/cvg/nice-slam#replica-1). It contains rendered trajectories using the mesh models provided by the original Replica datasets.

#### ScanNet
For ScanNet, please follow the instructions in [ScanNet](https://github.com/ScanNet/ScanNet).

### Environment Setup
Build docker image and create container:

```bash
./docker/build.sh
./docker/start.sh <path_to_data_folder>
./docker/into.sh
```

Install **bbq** library, call this once for container:
```bash
pip install -e .
```

### Run BBQ

First, build 3D scene representation. Check config before run. Inside container call script:

```python
python3 main.py examples/configs/replica_room0.yaml #Replica
python3 main.py examples/configs/scannet_scene0011_00.yaml #ScanNet
```

Construct 3D scene graph and enter natural language query. Inside container call script:
```python
python3 query.py examples/scenes/replica_room0.json #Replica
python3 query.py examples/scenes/scannet_scene0011_00.json #ScanNet
```

## Acknowledgement
We base our work on the following paper: [ConceptGraphs](https://github.com/concept-graphs/concept-graphs).

## Citation
If you find this work helpful, please consider citing our work as:
```
@misc{linok2024barequeriesopenvocabularyobject,
      title={Beyond Bare Queries: Open-Vocabulary Object Retrieval with 3D Scene Graph}, 
      author={Sergey Linok and Tatiana Zemskova and Svetlana Ladanova and Roman Titkov and Dmitry Yudin},
      year={2024},
      eprint={2406.07113},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.07113}, 
}
```

## Contact
Please create an issue on this repository for questions, comments and reporting bugs. Send an email to [Linok Sergey](linok.sa@phystech.edu) for other inquiries.
