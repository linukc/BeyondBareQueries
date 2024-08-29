import os

import wget
import torch
torch.set_grad_enabled(False)
import numpy as np
from loguru import logger
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAMGenerator:
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_PATH, "sam_vit_h_4b8939.pth")

    def __init__(self, weights_path):
        if not os.path.isfile(self.SAM_CHECKPOINT_PATH):
            try:
                self.SAM_CACHE_PATH = weights_path if weights_path else self.SAM_CACHE_PATH
                logger.info(f"Downloading SAM checkpoint to a {self.SAM_CACHE_PATH}.")
                wget.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", out=self.SAM_CACHE_PATH)
            except:
                raise RuntimeError("Can't load weights for SAM vit-h.")

        sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )

    def __call__(self, image):
        """
        image: [H, W, 3] numpy array
        """
        results = self.mask_generator.generate(image)
        masks = []
        xyxy = []
        conf = []
        for r in results:
            masks.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        masks = np.array(masks)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return xyxy, masks, conf
