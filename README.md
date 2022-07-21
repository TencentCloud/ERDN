### ERDN: Equivalent Receptive Field Deformable Network for Video Deblurring (ECCV 2022)

> [[Paper]]
>
> Bangrui Jiang, Zhihuai Xie, Zhen Xia, Songnan	Li, Shan Liu
>
> Tencent Media Lab, Shenzhen, China


## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux

## Installation

Clone repo

```bash
git clone xxx
cd ERDN
```

Install dependent packages

- torchvision
- tqdm
- imageio
- numpy
- opencv-python

We use [Deformable-Convolution-v2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0)
and install as follows

```bash
cd dcn
bash make.sh
cd ..
```

## Data Preparation
We use [DVD](https://github.com/shuochsu/DeepVideoDeblurring) for training and testing.
The dataset can be download as follows
```bash
wget http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip
unzip DeepVideoDeblurring_Dataset.zip
```
The data should be placed according to the following format
```
|--DVD
    |--Train
        |--blur 
            |--video 1
                |--frame 1
                |--frame 2
                    :  
            |--video 2
                :
            |--video n
        |--gt
            |--video 1
                |--frame 1
                |--frame 2
                    :
            |--video 2
                :
            |--video n
    |--Test
        |--blur
            |--video 1
                :
        |--gt
            |--video 1
                :
```

We provide preprocess script for DVD dataset
```
python script/arrange.py --data_path path_to_origin_DVD_dataset --out_path path_to_DVD
```


## Quick Inference

Download pre-trained models from [DVD](https://pan.baidu.com/s/1ZJrcGvolYoeianZwBI3DzQ) (key: csig).

Run following command for quick inference.

```bash
python inference.py \
--data_path path_to_DVD \
--model_path path_to_model \
--result_path path_to_save_result \ 
--save_image whether_to_save_image
```

## Training

The training script will be released soon.
