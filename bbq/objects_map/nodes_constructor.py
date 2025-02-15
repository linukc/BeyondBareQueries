import os
import gzip
import pickle

import torch
torch.set_grad_enabled(False)
from loguru import logger

from bbq.objects_map import ObjectsAssociator, DetectionsAssembler, \
    create_object_masks, describe_objects
from bbq.objects_map.utils import MapObjectList, merge_objects, postprocessing
from bbq.models import DINOFeaturesExtractor, ClassAgnosticMaskGenerator


class NodesConstructor:
    def __init__(self, config):
        self.config = config
        self.objects = MapObjectList()

        self.mask_generator = ClassAgnosticMaskGenerator(
            **config["mask_generator"])
        self.features_generator = DINOFeaturesExtractor(
            **config["dino_generator"])
        self.detections_assembler = DetectionsAssembler(
            **config["detections_assembler"])
        self.objects_mapper = ObjectsAssociator(
            **config["objects_associator"])

    def integrate(self, step_idx, frame, save_path=False):
        color, depth, intrinsics, pose = frame
        # generate class-agnostic masks
        masks_result = self.mask_generator(color)
        # generate DINO features
        descriptors = self.features_generator(color)
        # aggregate information about detected objects
        detected_objects, segmentation_vis = self.detections_assembler(
            step_idx, color, depth, intrinsics, pose, masks_result, descriptors)

        if len(detected_objects) == 0 and len(self.objects) != 0:
            logger.debug("no detected objects")
            return
        if len(self.objects) == 0:
            # add all detections to the map
            for i in range(len(detected_objects)):
                self.objects.append(detected_objects[i])
            logger.debug(f"Initialize {len(detected_objects)} detections as objects")

        # objects accumulation
        self.objects = self.objects_mapper(detected_objects, self.objects)
        if save_path:
            results = {'objects': self.objects.to_serializable()}
            with gzip.open(os.path.join(save_path, f"frame_{step_idx}_objects.pkl.gz"), "wb") as f:
                pickle.dump(results, f)
        return segmentation_vis
            
    def postprocessing(self):
        self.objects = postprocessing(self.objects, self.config)

    def project(self, poses, intrinsics):
        self.objects = create_object_masks(
            self.objects, poses, intrinsics,
            self.config["projector"]["num_views"],
            self.config["projector"]["top_k"],
            (self.config["projector"]["desired_height"], 
             self.config["projector"]["desired_width"])
        )

    def describe(self, colors):
        return describe_objects(self.objects, colors)
