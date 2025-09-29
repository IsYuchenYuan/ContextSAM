# ContextSAM

## Introduction

Welcome to the official implementation code for "Context-SAM: Towards Training-Free and Automatic 3D Medical Image Segmentation from In-Context Examples"

<!-- [**Arxiv Version**](https://arxiv.org/abs/2403.07636) -->

This work leverages SAM2 to perform in-context segmentation for 3D medical data. Our ContextSAM achieves the state-of-the-art performance across 6 datasets spanning different imaging modalities (e.g., CT, multi-sequence MRI, Ultrasound video) and diverse anotomical structures (e.g., abdominal organs, heart, knee).


<!-- ## 📝 Citation

If you find our work useful, please cite our paper.

```
@inproceedings{phan2024decomposing,
  title={Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Pre-training Framework},
  author={Phan, Vu Minh Hieu and Xie, Yutong and Qi, Yuankai and Liu, Lingqiao and Liu, Liyang and Zhang, Bowen and Liao, Zhibin and Wu, Qi and To, Minh-Son and Verjans, Johan W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11492--11501},
  year={2024}
} -->


## 🚀 Library Installation

Please run the following installation:
```
pip install -r requirements.txt
```


## 📦 Downstream datasets:
Links to download downstream datasets are:
* [CHAOS-MRI](https://chaos.grand-challenge.org)
* [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
* [WORD](https://github.com/HiLab-git/WORD)
* [SKI10](https://ski10.grand-challenge.org)
* [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)
* [MMWHS](https://zmiclab.github.io/zxh/0/mmwhs/)


## Example Structure
The support and query volume lists of each dataset are organised in the following way
```
ContextSAM/
├── data/
│   ├── Test_organ/
│       ├── acdc/
│           ├── query_img_list.txt
│           ├── query_lb_list.txt
│           ├── support_img_list.txt
│           ├── support_lb_list.txt
│       ├── word
│           ├── ...
│       ├── ...
└── ...
```


## 🌟 Quick Start:
* **Inference:**
```
python test_by_organ_sim.py
```
   


## 🙏 Acknowledgement
Our code is built upon https://github.com/facebookresearch/sam2 and https://github.com/ImprintLab/Medical-SAM2/tree/main. We thank the authors for open-sourcing their code.

Feel free to reach out if you have any questions or need further assistance! Contact me: ycyuan22@cse.cuhk.edu.hk
