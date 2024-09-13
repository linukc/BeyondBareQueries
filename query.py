import os
import random
import argparse

import torch
import numpy as np
from loguru import logger

from bbq.grounding import Llama3


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
    logger.info("Loading model")
    llm = Llama3(args.model_path)
    llm.set_scene(args.scene_file)

    while True:
        logger.info("Write user query:")
        user_query = input()

        logger.info("Selecting relevant nodes")
        related_objects = llm.select_relevant_nodes(user_query)
        logger.info("Selecting reffered object")
        full_answer, _ = llm.select_referred_object(user_query, related_objects)
        logger.info(full_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_file",
                        help="json where the objects bboxes and descriptions are stored.")
    parser.add_argument("--model_path",
                        help="file with pretrained LLM weights are stored.")
    args = parser.parse_args()

    set_seed()
    main(args)