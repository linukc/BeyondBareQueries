import os
import sys
import argparse

import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger

from bbq.datasets import get_dataset
from bbq.objects_map import NodesConstructor

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


class TqdmLoggingHandler:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message, **kwargs):
        if message.strip() != "":
            tqdm.write(message, end="")

    def flush(self):
        pass

def set_seed(seed: int=18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def main(args):
    with open(args.config_path) as file:
        config = yaml.full_load(file)
    logger.info(f"Parsed arguments. Utilizing config from {args.config_path}.")

    nodes_constructor = NodesConstructor(config["nodes_constructor"])
    rgbd_dataset = get_dataset(config["dataset"])

    # See Section 3.1
    logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
    for step_idx in tqdm(range(len(rgbd_dataset))):
        frame = rgbd_dataset[step_idx]
        nodes_constructor.integrate(step_idx, frame)
    torch.cuda.empty_cache()

    # See Section 3.2
    logger.info('Finding 2D view to caption 3D objects.')
    nodes_constructor.project(
        poses=rgbd_dataset.poses,
        intrinsics=rgbd_dataset[0][1]
    )
    torch.cuda.empty_cache()

    # See Section 3.3
    logger.info('Captioning 3D objects.')
    nodes = nodes_constructor.describe(
        colors=rgbd_dataset.color_paths
    )
    torch.cuda.empty_cache()

    logger.info('Saving results in json file.')
    with open(os.path.join(config["output_path"], config["output_name"]), 'w') as f:
        json.dump(nodes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Build 3D scene object map.
        For more information see Sec. 3.1 - 3.3.""")
    parser.add_argument("--config_path", default=r"examples/configs/replica_room0.yaml",
                        help="see example in default path")
    parser.add_argument("--logger_level", default="INFO")
    args = parser.parse_args()

    # Remove the default handler
    logger.remove()
    # Add a custom handler with tqdm support and set the level
    logger.add(TqdmLoggingHandler(), level=args.logger_level, colorize=True)

    set_seed()
    main(args)
