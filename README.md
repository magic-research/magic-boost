# Magic-Boost: Boost 3D Generation with Multi-View Conditioned Diffusion
Fan Yang, Jianfeng Zhang, Yichun Shi, Bowen Zhang, Chenxu Zhang, Huichao Zhang, Xiaofeng Yang, Xiu Li, Jiasheng Feng, Guosheng Lin

### [Project page](https://magic-research.github.io/magic-boost/) |  [Paper](https://arxiv.org/abs/2404.06429)

<div align="center">
  <img width="800" src="assets/teaser.png">
</div>

## Installation 

This part is the same as original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio) or [ImageDream-threestudio](https://github.com/bytedance/ImageDream). Skip it if you already have installed the environment.

## Pretrained weights
Clone the modelcard on the [Huggingface MagicBoost Model Page](https://huggingface.co/yyyfan/magic-boost/) and put the .pt files under  ```./extern/MVC/checkpoints/``` 

## Quick Start

### 1. Get the multi-view images and coarse meshes.
We are not able to open the reproduced Instant3D model currently. However, we provide two ways to use our model:

* #### Instant3D
  We provide pre-computed four view images and extracted meshes from the reproduced instant3d, which allows users to do a simple test of the model. 
  Please download the pre-computed images and meshes from [Huggingface MagicBoost Demo Page](https://huggingface.co/datasets/yyyfan/magic-boost-demo) and put the images and meshes into ```./load/mv_instant3d``` and ```./load/mesh_instant3d```

  <video src="./assets/demo_instant3d.mp4" controls="controls" width="500px"></video>

* #### InstantMesh
  Thanks to the open project [InstantMesh](https://github.com/TencentARC/InstantMesh), our model now supports using InstantMesh as a base model. Please install InstantMesh following the open repo [InstantMesh](https://github.com/TencentARC/InstantMesh) and run commands 
  ``` python
  python run.py configs/instant-mesh-large.yaml path_to_image --save_video
  ``` 
  to get the multi-view images and meshes.
  We provide a script  to preprocess the multi-view images into the right format. Simply run 
  ```
  python ./load/prepare_instantmesh.py --mvimage_path [path_to_instantmesh_output/images]
  ```
  Put the final mv images and the meshes into ```./load/mv_instantmesh``` and ```./load/mesh_instantmesh```

  <video src="./assets/demo_instantmesh.mp4" controls="controls" width="500px"></video>

### 2. Convert Mesh into Nerf
We first convert the coarse mesh into Nerf for differentiable rendering. To convert the mesh into Nerf, simply run 
```
python launch.py --gpu 0 --config ./configs/mesh-fitting-[instant3d/instantmesh].yaml --train tag=[name] system.guidance.geometry.shape_init=[target_obj_path]
```

**Recommend:** We provide a script to generate the commends automaicly, simply run 
```
python threestudio/scripts/batch_meshfitting.py --mode [instant3d/instantmesh]

bash run_mesh_fitting_scripts_[mode].sh
```


### 3. Refine
To refine the converted Nerf, simply run 
```
export PYTHONPATH=$PYTHONPATH:./extern/MVC

python launch.py --gpu 0 --config ./configs/refine_[instant3d/instantmesh].yaml --train tag=[name] system.prompt_processor.prompt="a " data.mvcond_dir=[multi-view_image_path] data.image_path=[input_image_path] system.weights=[ckpts_from_last_step]
```

**Recommend:** We also provide a script to automaticly search the ckpts and write the commands, simply run 
```
python threestudio/scripts/batch_refine.py --mode [instant3d/instantmesh] 

bash run_refine_[mode].sh
```

### Notes:
- We use batchsize 4 as default in our experiments, which would need an A100 GPU to do the computation in the refinement stage. To use the model with less VRAM, please adjust ```data.random_camera.batchsize``` in the config file to be lower. However, this may lead to slightly degraded results compared to batchsize of 4. Increasing the total refinement steps may help to address the problem and get a better result.  

- For diffusion only model, refer to subdir ```./extern/MVC/```. Check ```./extern/MVC/README.md``` for instruction.

## Acknowledgements
This code is built based on several open repos including [threestudio](https://github.com/threestudio-project/threestudio), [MVDream](https://github.com/bytedance/MVDream-threestudi), [ImageDream-threestudio](https://github.com/bytedance/ImageDream), [InstantMesh](https://github.com/TencentARC/InstantMesh) and [Dreamcraft3d](https://github.com/deepseek-ai/DreamCraft3D). We sincerely thank the authors of these projects for their excellent contributions to 3D generation. 

## Citing
If you find Magic-Boost helpful, please consider citing:

``` bibtex
@article@misc{yang2024magicboost,
      title={Magic-Boost: Boost 3D Generation with Mutli-View Conditioned Diffusion}, 
      author={Fan Yang and Jianfeng Zhang and Yichun Shi and Bowen Chen and Chenxu Zhang and Huichao Zhang and Xiaofeng Yang and Jiashi Feng and Guosheng Lin},
      year={2024},
      eprint={2404.06429},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```