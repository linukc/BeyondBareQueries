import os
import gzip
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import cv2
import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import imageio

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")

def main(args):
    hash = datetime.now()
    with open(args.config_path) as file:
        config = yaml.full_load(file)
    if args.save_path:
         os.makedirs(args.save_path, exist_ok=True)
         with gzip.open(os.path.join(args.save_path, "meta.pkl.gz"), "wb") as file:
            pickle.dump({"config": config}, file)
        
    logger.info(f"Parsed arguments. Utilizing config from {args.config_path}.")

    nodes_constructor = NodesConstructor(config["nodes_constructor"])
    #rgbd_dataset = get_dataset(config["dataset"])
    # See Section 3.1
    logger.info("Iterating over RGBD sequence to accumulate 3D objects.")
    color_path = "datasets/room0/color/2025-02-10-120131_1.jpg"
    #depth_path = self.depth_paths[index]
    color = np.asarray(imageio.imread(color_path), dtype=float)
    print(color.shape)
    color = cv2.resize(
        color,
        (1200, 680),
        interpolation=cv2.INTER_LINEAR,
    )
    #color = rgbd_dataset._preprocess_color(color) # resize
    color = torch.from_numpy(color)
    print(color.shape)

    
    color = color.to("cuda").type(torch.float)
    
    print(args.save_path)
    frame = (color, None, None, None)
    nodes_constructor.integrate(0, frame,
        args.save_path)

    torch.cuda.empty_cache()
    #nodes_constructor.postprocessing()
    torch.cuda.empty_cache()
    if args.save_path:
        results = {'objects': nodes_constructor.objects.to_serializable()}
        with gzip.open(os.path.join(args.save_path,
            f"frame_last_objects.pkl.gz"), "wb") as f:
                pickle.dump(results, f)

    # See Section 3.3
    logger.info('Captioning 3D objects.')
    nodes = nodes_constructor.describe(
        colors=[color_path]
    )
    torch.cuda.empty_cache()

    logger.info('Saving objects.')
    os.makedirs(config["nodes_constructor"]["output_path"], exist_ok=True)
    results = {'objects': nodes_constructor.objects.to_serializable()}
    with gzip.open(os.path.join(
        config["nodes_constructor"]["output_path"],
        hash.strftime("%m.%d.%Y_%H:%M:%S_") + config["nodes_constructor"]["output_name_objects"]),
        'wb') as file:
            pickle.dump(results, file)

    logger.info('Saving graph nodes in json file.')
    os.makedirs(config["nodes_constructor"]["output_path"], exist_ok=True)
    with open(os.path.join(
        config["nodes_constructor"]["output_path"],
        hash.strftime("%m.%d.%Y_%H:%M:%S_") + config["nodes_constructor"]["output_name_nodes"]),
        'w') as f:
            json.dump(nodes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Build 3D scene object map.
        For more information see Sec. 3.1 - 3.3.""")
    parser.add_argument("--config_path", default=r"examples/configs/replica_room0.yaml",
                        help="see example in default path")
    parser.add_argument("--logger_level", default="INFO")
    parser.add_argument("--save_path", default=None,
                        help="folder to save all steps to visualize mapping process")
    args = parser.parse_args()

    logger.remove()
    logger.add(TqdmLoggingHandler(), level=args.logger_level, colorize=True)

    set_seed()
    main(args)
