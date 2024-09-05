import os
import gzip
import pickle
import argparse

import imageio
import numpy as np
import open3d as o3d

from bbq.datasets import get_dataset
from bbq.objects_map.utils import MapObjectList


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def main(args):
    meta_path = os.path.join(args.animation_folder, "meta.pkl.gz")
    with gzip.open(meta_path, "rb") as file:
        meta_info = pickle.load(file)
    config = meta_info["config"]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name = "Mapping",
        width = config["dataset"]["desired_width"],
        height = config["dataset"]["desired_height"],
        visible=True
    )
    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(10)
    camera_param = view_ctrl.convert_to_pinhole_camera_parameters()

    result_frames = []
    rgbd_dataset = get_dataset(config["dataset"])
    for step_idx, frame in enumerate(rgbd_dataset):
        color, _, _, pose = frame

        # load the mapping results up to this frame
        with gzip.open(os.path.join(
            args.animation_folder, f"frame_{step_idx}_objects.pkl.gz"), "rb") as file:
                frame_objects = pickle.load(file)

        frame_objects_list = MapObjectList()
        frame_objects_list.load_serializable(frame_objects["objects"])

        # first render the objects in RGB color
        if step_idx > 0:
            vis.clear_geometries()
        pcds = frame_objects_list.get_values("pcd")
        bboxes = frame_objects_list.get_values("bbox")
        for geom in pcds + bboxes:
            vis.add_geometry(geom)
        camera_param.extrinsic = np.linalg.inv(to_numpy(pose))
        view_ctrl.convert_from_pinhole_camera_parameters(camera_param)
        view_ctrl.camera_local_translate(forward=-0.4, right=0.0, up=0.0)
        vis.poll_events()
        vis.update_renderer()
        render_rgb = vis.capture_screen_float_buffer(True)
        render_rgb = np.asarray(render_rgb)

        # save frames
        image_stack = np.concatenate([color, (render_rgb * 255).astype(np.uint8)], axis=1)
        result_frames.append(image_stack)

    vis.destroy_window()
    imageio.mimwrite(args.video_save_path, result_frames, fps=float(args.fps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animation_folder",
                        help="folder where the objects of the mapping process are stored.")
    parser.add_argument("--fps", default=5)
    parser.add_argument("--video_save_path",
                        default="output.mp4")
    args = parser.parse_args()
    main(args)
