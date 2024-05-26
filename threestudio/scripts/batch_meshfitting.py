import os
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--mode",
    type=str,
    help="select instant3d or instantmesh",
    )
    args = parser.parse_args()
    
    mode = args.mode

    if mode == "instant3d":
        image_folder = "mv_instant3d"
        mesh_folder = "mesh_instant3d"
        config = "mesh-fitting-instant3d.yaml"
    elif mode == "instantmesh":
        image_folder = "mv_instantmesh"
        mesh_folder = "mesh_instantmesh"
        config = "mesh-fitting-instantmesh.yaml"
    else:
        raise NotImplementedError

    files = os.listdir(f'./load/{image_folder}')
    files = [f'./load/{mesh_folder}/' + file for file in files]

    with open(f"run_mesh_fitting_scripts_{mode}.sh", "w") as f:
        for file in files:
            name = os.path.basename(file).split("/")[-1]
            obj_path = file + '.obj'

            cl = f"python launch.py --gpu 0 --config ./configs/{config} --train tag=\"{name}\" system.guidance.geometry.shape_init={obj_path}"
            f.write(cl + '\n')
