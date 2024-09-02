import cv2
import torch
import numpy as np
import open3d as o3d
from loguru import logger
import torch.nn.functional as F

from bbq.objects_map.utils import DetectionList
from bbq.objects_map.utils import process_pcd, get_bounding_box


def from_intrinsics_matrix(K):
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy

def to_scalar(d):
    if isinstance(d, float):
        return d
    
    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()
    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")

class DetectionsAssembler:
    def __init__(self, mask_conf_threshold,
                       mask_area_threshold,
                       max_bbox_area_ratio,
                       min_points_threshold,
                       downsample_voxel_size,
                       dbscan_remove_noise,
                       dbscan_eps,
                       dbscan_min_points,
                       image_area,
                **kwargs):
        self.mask_conf_threshold = mask_conf_threshold
        self.mask_area_threshold = mask_area_threshold
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_points_threshold = min_points_threshold
        self.downsample_voxel_size = downsample_voxel_size
        self.dbscan_remove_noise = dbscan_remove_noise
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.image_area = image_area

    def __call__(self, step_idx, color, depth, intrinsics, pose, masks_result, descriptors):
        detection_list = DetectionList()

        # filter low confidence proposals
        idx_to_save = []
        n_masks = len(masks_result["xyxy"])
        for mask_idx in range(n_masks):
            if masks_result['confidence'][mask_idx] < self.mask_conf_threshold:
                logger.debug(f"skipping mask with a low confidence. idx = {mask_idx}")
            else:
                idx_to_save.append(mask_idx)
        masks_result["mask"] = np.take(masks_result["mask"], idx_to_save, axis=0)
        masks_result["xyxy"] = np.take(masks_result["xyxy"], idx_to_save, axis=0)
        masks_result["confidence"] = np.take(masks_result["confidence"], idx_to_save, axis=0)
        
        # compute the containing relationship among all detections and subtract fg from bg objects
        masks_result['mask'] = self.mask_subtract_contained(masks_result['xyxy'], masks_result['mask'])

        # iterate over objects
        n_masks = len(masks_result['xyxy'])
        for mask_idx in range(n_masks):
            mask = masks_result['mask'][mask_idx]
            if mask.sum() < max(self.mask_area_threshold, 10):
                logger.debug(f"Skipping: mask is too small for idx = {mask_idx}")
                continue
            
            # skip small outliers
            x1, y1, x2, y2 = masks_result['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area > self.max_bbox_area_ratio * self.image_area:
                logger.debug(f"""Skipping object with area {bbox_area} > {self.max_bbox_area_ratio} * {self.image_area}.
                             idx = {mask_idx}""")
                continue

            # create object pcd
            camera_object_pcd = self.create_object_pcd(color, mask, depth, intrinsics)
            if len(camera_object_pcd.points) < max(self.min_points_threshold, 5):
                logger.debug(f"""Skipping: num points {camera_object_pcd.points}
                             < min points {max(self.min_points_threshold, 5)}""")
                continue
            elif len(camera_object_pcd.points) < max(2 * self.min_points_threshold, 5):
                logger.debug(f"Warning: few points number for {mask_idx} - less than 2 * MIN_POINTS_THRESHOLD")
            global_object_pcd = camera_object_pcd.transform(pose.cpu().numpy())

            # filter noise
            global_object_pcd, perc_preserve = process_pcd(global_object_pcd, self.downsample_voxel_size,
                self.dbscan_remove_noise, self.dbscan_eps, self.dbscan_min_points, run_dbscan=True)
            if not perc_preserve:
                logger.debug(f"Skipping: dbscan find only noise for det {mask_idx}")
                continue
            elif perc_preserve <= 0.9:
                logger.debug(f"Skipping: dbscan most common cluster is not a dominant with {perc_preserve}%")
                continue

            # filter small volumes
            pcd_bbox = get_bounding_box(global_object_pcd)
            pcd_bbox.color = [0, 1, 0]
            if pcd_bbox.volume() < 1e-6:
                logger.debug("Skipping: bbox volume after downsample is very low")
                continue
            
            # calculate object descriptor
            local_mask = masks_result["mask"][mask_idx]
            local_mask = torch.tensor(local_mask)
            local_mask = F.interpolate(local_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(descriptors.shape[0], descriptors.shape[1]),
                mode='nearest').squeeze(0).squeeze(0).bool()
            loc_descriptor = descriptors[local_mask].mean(dim=0, keepdim=True)
            if torch.any(torch.isnan(descriptors)):
                if not torch.any(local_mask):
                    logger.debug(f"Downsampled mask {mask_idx} does not contain any True values")
                continue
            
            # form object dict
            detected_object = {
                'pcd': global_object_pcd, # pointcloud
                'bbox': pcd_bbox, # bbox
                'descriptor': loc_descriptor, # descriptor  # [1, d]
                'num_detections': 1, # number of detections (for filtering)
                'id': {step_idx}, # detection frame idx (for projection)
            }
            detection_list.append(detected_object)

        return detection_list

    def mask_subtract_contained(self, xyxy, mask, th1=0.8, th2=0.7):
        '''
        Compute the containing relationship between all pair of bounding boxes.
        For each mask, subtract the mask of bounding boxes that are contained by it.
        
        Args:
            xyxy: (N, 4), in (x1, y1, x2, y2) format
            mask: (N, H, W), binary mask
            th1: float, threshold for computing intersection over box1
            th2: float, threshold for computing intersection over box2
            
        Returns:
            mask_sub: (N, H, W), binary mask
        '''

        # Get areas of each xyxy
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

        # Compute intersection boxes
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
        inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

        # Compute areas of intersection boxes
        inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
        inter_over_box1 = inter_areas / areas[:, None] # (N, N)

        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
        inter_over_box2 = inter_over_box1.T # (N, N)

        # if the intersection area is smaller than th2 of the area of box1, 
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
        contained_idx = contained.nonzero() # (num_contained, 2)
        mask_sub = mask.copy() # (N, H, W)
    
        kernel = np.ones((5, 5), np.uint8)
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])
            ### remove edge noise after substraction
            mask_ = cv2.numpy.asmatrix(mask_sub[contained_idx[0][i]].astype(np.uint8))
            mask_ = cv2.erode(mask_, kernel, iterations=2) 
            mask_ = cv2.dilate(mask_, kernel, iterations=2)
            mask_sub[contained_idx[0][i]] = np.asarray(mask_, bool)
        return mask_sub
    
    def create_object_pcd(self, color, mask, depth_array, cam_K):
        depth_array = depth_array[..., 0].cpu().numpy()
        # Also remove points with invalid depth values
        mask = np.logical_and(mask, depth_array > 0)

        if mask.sum() == 0:
            pcd = o3d.geometry.PointCloud()
            return pcd
        
        fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
        height, width = depth_array.shape
        x = np.array(np.arange(0, width, 1.0))
        y = np.array(np.arange(0, height, 1.0))
        u, v = np.meshgrid(x, y)
        
        # Apply the mask, and unprojection is done only on the valid points
        masked_depth = depth_array[mask] # (N, )
        u = u[mask] # (N, )
        v = v[mask] # (N, )

        # Convert to 3D coordinates
        x = (u - cx) * masked_depth / fx
        y = (v - cy) * masked_depth / fy
        z = masked_depth

        # Stack x, y, z coordinates into a 3D point cloud
        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)
        
        # Perturb the points a bit to avoid colinearity
        points += np.random.normal(0, 4e-3, points.shape)

        if points.shape[0] == 0:
            raise RuntimeError("zero points pcd")

        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # colors = color.numpy()[mask] / 255.0
        obj_color = np.random.rand(3)
        colors = np.full(points.shape, obj_color)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
