import torch
import pytorch3d.ops as ops
import torch.nn.functional as F


def compute_spatial_similarities(detection_list, objects) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects

    Args:
        detection_list: a list of M detections (DetectionList)
        objects: a list of N objects in the map (MapObjectList)
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    return spatial_sim

def compute_visual_similarities(detection_list, objects, spatial_sim) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections (DetectionList)
        objects: a list of N objects in the map (MapObjectList)
        spatial_sim: M x N torch sim matrix
    Returns:
        A MxN tensor of visual similarities
    '''
    visual_sim = torch.zeros((len(detection_list), len(objects))).fill_(float('-inf'))
    for i, det in enumerate(detection_list):
        det_features = det["descriptor"] # [1, D]
        for j, obj in enumerate(objects):
            if spatial_sim[i, j] != float('-inf'):
                obj_features = obj["descriptor"]
                sim = F.cosine_similarity(det_features.cuda(), obj_features.cuda(), dim=-1).cpu()
                visual_sim[i][j] = sim

    return visual_sim # (M, N)

def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    '''
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou
    
def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    '''
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention. 
    
    bbox: (N, 8, D)
    
    returns: (N, 8, D)
    '''
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
    
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    
    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)
    
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1) # shape: (N, 8, D)
    
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox
