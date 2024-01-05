<h2 align="center">LoveNAS: Hierarchically Search for Adaptive Neural Network in Remote Sensing Land-cover Mapping</h2>


<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Yuting Wan, Ailong Ma and Liangpei Zhang</h5>

This is an official implementation of LoveNAS.
---------------------

<div align="center">
  <img width="100%" src="https://github.com/Junjue-Wang/resources/blob/main/LoveNAS/framework.png?raw=true">
</div>


## Environments:
- pytorch >= 1.11.0
- python >=3.6

```bash
pip install --upgrade git+https://gitee.com/zhuozheng/ever_lite.git@v1.4.5
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```
The Swin-Transformer pretrained weights can be prepared following [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin).

### Search architecture 
```bash
bash ./scripts/nas_loveda.sh
```

### Train model
The searched architectures and transferred encoder weights should be downloaded.
```bash 
bash ./scripts/train_loveda.sh
```

### Predict test results
The searched architectures and LoveNAS model weights should be downloaded.
Submit the test results to [LoveDA Semantic Segmentation Challenge](https://codalab.lisn.upsaclay.fr/competitions/421) to get scores.
```bash 
bash ./scripts/submit_loveda.sh
```

### LoveDA
The LoveDA dataset can be downloaded [here](https://doi.org/10.5281/zenodo.5706578).

Submit the test results to [LoveDA Semantic Segmentation Challenge](https://codalab.lisn.upsaclay.fr/competitions/421) to get scores.

| Search-Config |     Backbone    | Train-Config | Params (M) |mIoU(%) | Download |
|:----------:|:---------------:|:------------:|:-------:|:-------:|:--------:|
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/nas_mobilenet_lovedecoder.py)           |   MobileNetV2   | [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/train_mobilenet_lovedecoder.py)     |  3.837      |  50.60  |   [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)   |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/nas_resnet_lovedecoder.py)           |    ResNet-50    | [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/train_resnet_lovedecoder.py)           | 30.491 |  52.34  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)         |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/nas_efnet_lovedecoder.py)           | EfficientNet-B3 |  [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/train_efnet_lovedecoder.py)      | 14.190     |  52.05  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)         |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/nas_swin_lovedecoder.py)           |    Swin-Base    |  [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/loveda/train_swin_lovedecoder.py)        | 92.435   |  53.76  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)         |

### FloodNet
The FloodNet dataset can be downloaded [here](https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH).

The train data should be prepared using [prepare_floodnet.py](https://github.com/Junjue-Wang/LoveNAS/blob/master/tools/prepare_floodnet.py).

| Search-Config |     Backbone    | Train-Config | Params (M) |mIoU(%) | Download |
|:----------:|:---------------:|:------------:|:-------:|:-------:|:--------:|
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/nas_mobilenet_lovedecoder.py)           |   MobileNetV2   | [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/train_mobilenet_lovedecoder.py)              | 12.072 |70.73  |   [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)   |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/nas_resnet_lovedecoder.py)           |    ResNet-50    | [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/train_resnet_lovedecoder.py)             | 38.457 |72.54  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)          |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/nas_efnet_lovedecoder.py)           | EfficientNet-B3 |  [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/train_efnet_lovedecoder.py)            | 18.851  |72.69  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)         |
| [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/nas_swin_lovedecoder.py)           |    Swin-Base    |  [config](https://github.com/Junjue-Wang/LoveNAS/blob/master/configs/lovenas/floodnet/train_swin_lovedecoder.py)            | 97.701 |73.79  | [log&ckpt](https://pan.baidu.com/s/1DLKiVhVQ7KpBDk6JXHi9BA?pwd=2333) (pwd:2333)         |


## Citation
If you use LoveNAS in your research, please cite the following papers.
```text
    @inproceedings{wang2021loveda,
        title={Love{DA}: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
        author={Junjue Wang and Zhuo Zheng and Ailong Ma and Xiaoyan Lu and Yanfei Zhong},
        booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
        editor = {J. Vanschoren and S. Yeung},
        year={2021},
        volume = {1},
        pages = {},
        url={https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf}
    }
    @article{wang2020rsnet,
      title={RSNet: The search for remote sensing deep neural networks in recognition tasks},
      author={Wang, Junjue and Zhong, Yanfei and Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      volume={59},
      number={3},
      pages={2520--2534},
      year={2020},
      publisher={IEEE}
    }
```
