import torch
torch.set_grad_enabled(False)
from loguru import logger

from bbq.objects_map import ObjectsAssociator, DetectionsAssembler, \
    create_object_masks, describe_objects
from bbq.objects_map.utils import MapObjectList, merge_objects
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

    def integrate(self, step_idx, frame):
        color, depth, intrinsics, pose = frame

        # generate class-agnostic masks
        masks_result = self.mask_generator(color)

        # generate DINO features
        descriptors = self.features_generator(color)

        # aggregate information about detected objects
        detected_objects = self.detections_assembler(
            step_idx, depth, intrinsics, pose, masks_result, descriptors)
        
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

        # postprocessing
        if step_idx > 0 and step_idx % self.config["merge_interval"] == 0:
            self.objects = merge_objects(self.objects,
                self.config["merge_objects_overlap_thresh"],
                self.config["merge_objects_visual_sim_thresh"],
                self.config["detections_assembler"]["downsample_voxel_size"])

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