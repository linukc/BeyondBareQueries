import torch
import numpy as np
import supervision as sv

from bbq.models.masks.sam import SAMGenerator
from bbq.models.masks.mobile_sam import MobileSAMGenerator


class ClassAgnosticMaskGenerator:
    def __init__(self, model, weights_path, **kwargs):
        if model == "SAM":
            self.model = SAMGenerator(weights_path)
        elif model == "MobileSAM":
            self.model = MobileSAMGenerator(weights_path)
        else:
            raise NotImplementedError
    
    def __call__(self, image):
        image = image.cpu().to(torch.uint8).numpy()
        xyxy, mask, conf = self.model(image)
        xyxy, mask, conf = xyxy.cpu().numpy(), mask.cpu().bool().numpy(), conf.cpu().numpy().flatten()
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros_like(conf).astype(int),
            mask=mask,
        )
        results = {
            "xyxy": detections.xyxy,
            "mask": detections.mask,
            "confidence":detections.confidence
        }
        return results
