# Magic-Boost: Boost 3D Generation with Multi-View Conditioned Diffusion
Fan Yang, Jianfeng Zhang, Yichun Shi, Bowen Zhang, Chenxu Zhang, Huichao Zhang, Xiaofeng Yang, Xiu Li, Jiasheng Feng, Guosheng Lin

### [Project page](https://magic-research.github.io/magic-boost/) |  [Paper](https://arxiv.org/abs/2404.06429)

#### Notes: 
* This repo inherit content from repos of [MagicBoost](https://github.com/magic-research/magic-boost)

* It only includes the diffusion model and 2D image generation. For 3D Generation, please check [Here](https://github.com/magic-research/magic-boost).

## Installation
Setup environment as in [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion) for this repo. You can set up the environment by installing the given requirements
``` bash
pip install -r requirements.txt
```

## Novel View Synthesize
Clone the modelcard on the [Huggingface MagicBoost Model Page](https://huggingface.co/yyyfan/magic-boost/) under ```./release_models/```

``` bash
bash extern/MVC/scripts/demo.sh
```

Tips
- We provide examples from Instant3d and InstantMesh for examples. To use otehr methods with different multi-view condition settings, try to modify the can_camera setting in line 149-162 in demo.py. 


## Acknowledgement
This repository is heavily based on [Imagedream](https://github.com/bytedance/ImageDream). We would like to thank the authors of these work for publicly releasing their code.

## Citation
If you find ImageDream helpful, please consider citing:

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
