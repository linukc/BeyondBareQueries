import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def create_object_masks(objects, poses, cam_K, num_views, top_k, image_shape):
    scene = o3d.t.geometry.RaycastingScene()
    mesh = {}
    cam_K = cam_K.cpu().numpy()[:3, :3, 0]

    for i, object in enumerate(tqdm(objects)):
        mesh[i] = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(object['pcd'], 0.1)
        mesh[i] = o3d.t.geometry.TriangleMesh.from_legacy(mesh[i])
        try:
            scene.add_triangles(mesh[i])
        except Exception as e:
            logger.critical(f"Error adding mesh for object {i}: {e}")

    for id_object, object in enumerate(tqdm(objects)):
        pixel_area = []
        masks = []

        obj_poses = []
        for idx in list(object["id"]):
            obj_poses.append(poses[idx][:3, 3].cpu().numpy().tolist())
        obj_poses = np.array(obj_poses)\
        
        if len(obj_poses) < num_views:
            top_indices = range(len(list(object["id"])))
        else:
            kmeans = KMeans(n_clusters=5, max_iter=500)
            kmeans.fit_predict(torch.tensor(obj_poses, device="cuda"))
            centers = kmeans.centroids.cpu().numpy()

            top_indices, _ = pairwise_distances_argmin_min(centers, obj_poses)

        for i in top_indices:
            idx = list(object["id"])[i]
            view_pose = poses[idx]
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=cam_K, 
                extrinsic_matrix=np.linalg.inv(view_pose), 
                width_px=image_shape[1], 
                height_px=image_shape[0])
            ans = scene.cast_rays(rays)
            view_mask = ans['geometry_ids'].numpy() == id_object 

            masks.append([view_mask])
            pixel_area.append([view_mask.sum()])
        
        conf = np.array(pixel_area).squeeze()
        idx_most_conf = np.argsort(conf)[::-1]
        assert top_k == 1
        idx_most_conf = idx_most_conf[:top_k][0]
        object["color_image_idx"] = list(object["id"])[idx_most_conf]
        object["mask"] = masks[idx_most_conf][0]

    return objects