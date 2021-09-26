# UTA
The PyTorch implementation of [RGB-D Salient Object Detection with Ubiquitous Target Awareness](https://ieeexplore.ieee.org/abstract/document/9529069).

NOTE: UTA is an extension work of [DASNet](https://github.com/iCVTEAM/DASNet).

## Prerequisites

- python=3.7
- pytorch==1.6.0
- torchvision=0.7.0
- apex==0.1
- opencv-python==4.2

## Dir

- res: resnet pre-trained models
- eval: test results
- data: datasets
- checkpoint: models

## Datasets

- [Train data](https://drive.google.com/file/d/1DTkjjPr7MdAxJpOQAjYLhdOFUaxFLSqb/view?usp=sharing)
- [Test data](https://drive.google.com/drive/folders/16Mozx_lxsjEgAK3JCjKkFmaSml8FFVFr?usp=sharing)
- [Test results](https://drive.google.com/drive/folders/1tqmzEC4KAPgvFUhY_i3W10fitXWhNGxC?usp=sharing)

## Model

- Download the [model](https://drive.google.com/file/d/1ZSx-UK8W3kPd-T60ukj-zNOPJUCs10Es/view?usp=sharing) into `checkpoint` folder.

## Train

```shell script
cd src
python train_UTA.py
```

## Test
```shell script
cd src
python test.py
```

## Evaluation
```shell
cd eval
matlab
main
```

## Citation
- If you find this work is helpful, please cite our paper
```
@ARTICLE{zhao2021UTA,
  author={Zhao, Yifan and Zhao, Jiawei and Li, Jia and Chen, Xiaowu},
  journal={IEEE Transactions on Image Processing}, 
  title={RGB-D Salient Object Detection With Ubiquitous Target Awareness}, 
  year={2021},
  volume={30},
  number={},
  pages={7717-7731},
  doi={10.1109/TIP.2021.3108412}}
```
## Reference
This project is based on the following implementations:
- [https://github.com/iCVTEAM/DASNet](https://github.com/iCVTEAM/DASNet)

