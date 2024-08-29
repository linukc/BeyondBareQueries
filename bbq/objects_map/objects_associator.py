from bbq.objects_map.utils import process_pcd, get_bounding_box, \
    compute_spatial_similarities, compute_visual_similarities, merge_obj2_into_obj1


class ObjectsAssociator:
    def __init__(self,
                 merge_det_obj_spatial_sim_thresh,
                 merge_det_obj_visual_sim_thresh,
                 downsample_voxel_size,
                 **kwargs):
        self.merge_det_obj_spatial_sim_thresh = merge_det_obj_spatial_sim_thresh
        self.merge_det_obj_visual_sim_thresh = merge_det_obj_visual_sim_thresh
        self.downsample_voxel_size = downsample_voxel_size

    def __call__(self, detected_objects, scene_objects):
        # compute spatial sim
        spatial_sim = compute_spatial_similarities(detected_objects, scene_objects)
        spatial_sim[spatial_sim <= self.merge_det_obj_spatial_sim_thresh] = float('-inf')

        # compute vis sim
        visual_sim = compute_visual_similarities(detected_objects, scene_objects, spatial_sim)
        visual_sim[visual_sim < self.merge_det_obj_visual_sim_thresh] = float('-inf')

        # merge det to objects
        scene_objects = self.merge_detections_to_objects(detected_objects, scene_objects,
            visual_sim, self.downsample_voxel_size)

        return scene_objects

    def merge_detections_to_objects(self, detected_objects, scene_objects, visual_sim, downsample_voxel_size):
        # Iterate through all detections and merge them into objects
        for i in range(visual_sim.shape[0]):

            # If not matched to any object, add it as a new object
            if visual_sim[i].max() == float('-inf'):
                scene_objects.append(detected_objects[i])

            # Merge with most similar existing object
            else:
                j = visual_sim[i].argmax()
                matched_det = detected_objects[i]
                matched_obj = scene_objects[j]
                merged_obj = merge_obj2_into_obj1(matched_obj, matched_det,
                    downsample_voxel_size, run_dbscan=False, are_objects=False)
                scene_objects[j] = merged_obj

        return scene_objects
