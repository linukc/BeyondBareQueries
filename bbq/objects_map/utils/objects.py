from collections import Counter

import faiss
import torch
import numpy as np
import open3d as o3d
from loguru import logger
import torch.nn.functional as F
from cuml import DBSCAN as cumlDBSCAN

from bbq.objects_map.utils.structures import MapObjectList


def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

def get_bounding_box(pcd):
    if len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            logger.debug(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()
    
def process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise=None,
            dbscan_eps=None, dbscan_min_points=None, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        
    if dbscan_remove_noise and run_dbscan:
        pcd, perc_preserve = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        return pcd, perc_preserve
        
    return pcd

def pcd_denoise_dbscan(pcd, eps=0.02, min_points=10):
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)

    cuml_dbscan = cumlDBSCAN(eps=eps, min_samples=min_points)
    trained_DBSCAN = cuml_dbscan.fit(obj_points)
    pcd_clusters = trained_DBSCAN.labels_

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    perc_to_preserve = None
    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            logger.debug("point cloud is too small")
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd

        perc_to_preserve = len(largest_cluster_points) / len(obj_points)
        
    return pcd, perc_to_preserve

def compute_overlap_matrix(objects, downsample_voxel_size):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    res = faiss.StandardGpuResources()
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(arr.shape[1])) for arr in point_arrays]
    #indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])
                overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap

def merge_overlap_objects(objects, overlap_matrix,
        merge_overlap_thresh, merge_visual_sim_thresh, downsample_voxel_size):

    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        sim_ = F.cosine_similarity(to_tensor(objects[i]['descriptor'].cuda()),
                                   to_tensor(objects[j]['descriptor'].cuda()), dim=-1).cpu()
        visual_sim = sim_.item()

        if ratio > merge_overlap_thresh:
            if visual_sim > merge_visual_sim_thresh:
                if kept_objects[j]:
                    objects[j] = merge_obj2_into_obj1(objects[j], objects[i],
                        downsample_voxel_size, run_dbscan=False, are_objects=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects

def merge_obj2_into_obj1(obj1, obj2, downsample_voxel_size, run_dbscan=False, are_objects=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''

    for k in obj1.keys():
        if k not in ['pcd', 'bbox', 'descriptor', 'mask', 'id']:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k] # num detections
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, descriptor, id are handled below
            continue

    # merge pcd and bbox
    color = np.array(obj1['pcd'].colors)[0]
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], downsample_voxel_size,
                            run_dbscan=run_dbscan)
    
    colors = np.full((len(obj1['pcd'].points), 3), color)
    obj1['pcd'].colors = o3d.utility.Vector3dVector(colors) 
    obj1['bbox'] = get_bounding_box(obj1['pcd'])
    obj1['bbox'].color = [0, 1, 0]

    # merge descriptors
    if not are_objects:
        obj1['descriptor'] = obj1['descriptor'] * 0.2 + obj2['descriptor'] * 0.8
    else:
        obj1['descriptor'] = (obj1['descriptor'] + obj2['descriptor']) / 2
    
    # update id list
    obj1['id'].update(obj2['id'])

    return obj1

def merge_objects(objects, merge_overlap_thresh, merge_visual_sim_thresh, downsample_voxel_size):
    if merge_overlap_thresh > 0:
        # Merge one object into another if the former is contained in the latter
        logger.debug("Start merging")
        logger.debug(f"Before merging: {len(objects)}")
        overlap_matrix = compute_overlap_matrix(objects, downsample_voxel_size)
        objects = merge_overlap_objects(objects, overlap_matrix,
            merge_overlap_thresh, merge_visual_sim_thresh, downsample_voxel_size)
        logger.debug(f"After merging: {len(objects)}")

    return objects
