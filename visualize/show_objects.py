import gzip
import pickle
import argparse

import numpy as np
import open3d as o3d

from bbq.objects_map.utils import MapObjectList


def main(args):
    with gzip.open(args.objects_path, "rb") as f:
        results = pickle.load(f)
    objects = MapObjectList()
    objects.load_serializable(results["objects"])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.objects_path, width=1280, height=720)

    pcds = objects.get_values("pcd")
    bboxes = objects.get_values("bbox")
    for i in range(len(bboxes)):
        vis.add_geometry(bboxes[i])
        pcd_i = pcds[i].voxel_down_sample(voxel_size=0.05)
        color_rand = np.random.rand((3))
        pcd_i.colors = o3d.utility.Vector3dVector(
            [list(color_rand) for _ in range(len(pcd_i.points))])
        vis.add_geometry(pcd_i)

    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 40.0
    render_option.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects_path",
                        help="path to .pkl.gz file")
    args = parser.parse_args()
    main(args)