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
        weights_folder = "mesh-fitting-nerf-instant3d"
        config = "refine_instant3d.yaml"
    elif mode == "instantmesh":
        image_folder = "mv_instantmesh"
        weights_folder = "mesh-fitting-nerf-instantmesh"
        config = "refine_instantmesh.yaml"
    else:
        raise NotImplementedError

    root_dir = f"./load/{image_folder}"
    dirs = os.listdir(root_dir)
    mvdirs = [root_dir + '/' + d for d in dirs]
    prompts = [f"a " for d in dirs]

    weights_root = f"./outputs/{weights_folder}"
    all_weights = os.listdir(weights_root)
    weights = []
    for d in dirs:
        pweights = []
        for w in all_weights:
            if d == w.split('@')[0]:
                pweights.append(w)
        if len(pweights) == 0:
            print(d)
            continue
        pweights = sorted(pweights, key=lambda x:x.split('@')[-1])
        weights.append(os.path.join(weights_root, pweights[-1]) + "/ckpts/last.ckpt")

    with open(f"run_refine_{mode}.sh", "w") as f:
        f.write("export PYTHONPATH=$PYTHONPATH:./extern/MVC\n")
        for file, prompt, weight in zip(mvdirs, prompts, weights):
            name = os.path.basename(file).split(".")[0]
            image_path = file + '/0.png'

            cl = f"python launch.py --gpu 0 --config ./configs/{config} --train tag=\"{name}\" system.prompt_processor.prompt=\"{prompt}\" system.prompt_processor.image_path={image_path} data.mvcond_dir={file} data.image_path={image_path} system.weights={weight}"
            f.write(cl + '\n')
