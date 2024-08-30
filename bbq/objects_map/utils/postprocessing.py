import numpy as np
from loguru import logger

from bbq.objects_map.utils.structures import MapObjectList
from bbq.objects_map.utils.objects import merge_obj2_into_obj1, \
    process_pcd, get_bounding_box, merge_objects, compute_overlap_matrix


def merge_objects_postprocessing(objects: MapObjectList, bool_matrix, downsample_voxel_size):
    x, y = bool_matrix.nonzero()
    bool_matrix = bool_matrix[bool_matrix]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, bool_matrix):
        if ratio :
            if kept_objects[j]:
                objects[j] = merge_obj2_into_obj1(
                    objects[j], objects[i], downsample_voxel_size, are_objects=True)
                kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    return objects

def denoise_objects(objects: MapObjectList, downsample_voxel_size, dbscan_remove_noise,
                    dbscan_eps, dbscan_min_points):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], downsample_voxel_size,
                                dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True)[0]
        if len(objects[i]['pcd'].points) < 20:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(objects[i]['pcd'])
        objects[i]['bbox'].color = [0, 1, 0]
    return objects

def postprocessing(objects, config):
    logger.info(f"Before postprocessing: {len(objects)} objects")

    logger.debug("Start filtering")
    logger.debug(f"Before: {len(objects)}")
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= config["postprocessing"]["obj_min_points"] and \
            obj['num_detections'] >= config["postprocessing"]["obj_min_detections"]:
                objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    logger.debug(f"After: {len(objects)}")

    logger.debug("Start denoising")
    logger.debug(f"Before: {len(objects)}")
    objects = denoise_objects(objects,
        config["detections_assembler"]["downsample_voxel_size"],
        True,
        config["detections_assembler"]["dbscan_eps"],
        config["detections_assembler"]["dbscan_min_points"])
    logger.debug(f"After: {len(objects)}")

    logger.debug("Start merging")
    logger.debug(f"Before: {len(objects)}")
    _objects_count = 0
    while (_objects_count != len(objects)):
        if _objects_count != 0:
            logger.debug("Repeating merging step")
        _objects_count = len(objects)
        objects = merge_objects(objects,
            config["objects_associator"]["merge_objects_overlap_thresh"],
            config["objects_associator"]["merge_objects_visual_sim_thresh"],
            config["detections_assembler"]["downsample_voxel_size"])
    logger.debug(f"After: {len(objects)}")

    logger.debug("Start spatial merging postprocess")
    overlap_matrix = compute_overlap_matrix(objects,
        config["detections_assembler"]["downsample_voxel_size"])
    matrix_bool = overlap_matrix > 0.3
    matrix_bool = matrix_bool * matrix_bool.T
    logger.debug(f"Before: {len(objects)}")
    objects = merge_objects_postprocessing(objects, matrix_bool,
        config["detections_assembler"]["downsample_voxel_size"])
    logger.debug(f"After: {len(objects)}")

    logger.info(f"After postprocessing: {len(objects)} objects")
    return objects