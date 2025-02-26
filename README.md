# CMT
Code release for the paper **CMT: Co-training Mean-Teacher for Unsupervised Domain Adaptation on 3D Object Detection**, ACM MM 2024. The codebase is still under updating.
<!-- CMT is an unsupervised domain adaption method on 3D object detection, which is accepted on ACM MM 2024. -->

<!-- > [**Co-training Mean-Teacher for Unsupervised Domain Adaptation on 3D Object Detection**] -->
<!-- (https://arxiv.org/abs/2304.09446)          -->
<!-- > Shijie Chen et al  -->

<!-- Copyright (C) 2024 Shijie Chen  -->

<!-- License: MIT for academic use. -->

<!-- Contact: Shijie Chen (chensj37@foxmail.com) -->

## Introduction
CMT is an unsupervised domain adaption method on 3D object detection.

## Model Zoo

<!-- ## Model Zoo -->
We provide pretrained models here for these task in [models](https://drive.google.com/drive/folders/1-GnctiUgJl69owILhWq65hHvvAsmfS8I?usp=drive_link).

### nuScenes -> KITTI TASK
|                                                                                                | Car@R40 |
|------------------------------------------------------------------------------------------------|:-------:|
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_dts/dts.yaml)                       |  72.1   | 
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn_dts/dts.yaml)                             |  74.5   |


### Waymo -> KITTI TASK

|                                                                                                 | Car@R40 |
|-------------------------------------------------------------------------------------------------|:-------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_dts/dts.yaml)                           |  68.1  |
| [PVRCNN](tools/cfgs/da-waymo-kitti_models/pvrcnn_dts/dts.yaml)                                  |  75.5  |

### Waymo -> nuScenes TASK
|                                                                         | Car@R40 |
|-------------------------------------------------------------------------|:-------:|
| [SECOND-IoU](tools/cfgs/da-waymo-nus_models/secondiou_dts/dts.yaml)     | 24.3   | 
| [PVRCNN](tools/cfgs/da-waymo-nus_models/pvrcnn_dts/dts.yaml)            | 26.4   |


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Usage

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## Citation

If you find this project useful in your research, please consider cite our paper.
```
@inproceedings{chen2024cmt,
  title={CMT: Co-training Mean-Teacher for Unsupervised Domain Adaptation on 3D Object Detection},
  author={Chen, Shijie and Zhuo, Junbao and Li, Xin and Liu, Haizhuang and Wang, Rongquan and Chen, Jiansheng and Ma, Huimin},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={4738--4747},
  year={2024}
}
```

## Acknowledgement

Our code is heavily based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [ST3D](https://github.com/CVMI-Lab/ST3D), [DTS](https://github.com/WoodwindHu/DTS). Thanks for their awesome codebase.
