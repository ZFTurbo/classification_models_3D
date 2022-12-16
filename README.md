# Classification models 3D Zoo - Keras and TF.Keras

This repository contains 3D variants of popular CNN models for classification like ResNets, DenseNets, VGG, etc. It also contains weights 
obtained by converting ImageNet weights from the same 2D models. 

This repository is based on great [classification_models](https://github.com/qubvel/classification_models) repo by [@qubvel](https://github.com/qubvel/)

### Architectures: 
- [VGG](https://arxiv.org/abs/1409.1556) [16, 19]
- [ResNet](https://arxiv.org/abs/1512.03385) [18, 34, 50, 101, 152]
- [ResNeXt](https://arxiv.org/abs/1611.05431) [50, 101]
- [SE-ResNet](https://arxiv.org/abs/1709.01507) [18, 34, 50, 101, 152]
- [SE-ResNeXt](https://arxiv.org/abs/1709.01507) [50, 101]
- [SE-Net](https://arxiv.org/abs/1709.01507) [154]
- [DenseNet](https://arxiv.org/abs/1608.06993) [121, 169, 201]
- [Inception ResNet V2](https://arxiv.org/abs/1602.07261)
- [Inception V3](http://arxiv.org/abs/1512.00567)
- [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNet v2](https://arxiv.org/abs/1801.04381)
- [EfficientNet](https://arxiv.org/abs/1905.11946) [B0, B1, B2, B3, B4, B5, B6, B7]
- [EfficientNet v2](https://arxiv.org/abs/2104.00298) [B0, B1, B2, B3, S, M, L]
- [ConvNeXt](https://arxiv.org/pdf/2201.03545.pdf)

### Installation 

`pip install classification-models-3D`

### Examples 

#### Loading model with `imagenet` weights:

```python

from classification_models_3D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(input_shape=(128, 128, 128, 3), weights='imagenet')
```

#### Create model examples:
- [tst_tfkeras.py](tst_tfkeras.py)
- [tst_tfkeras_special_cases.py](tst_tfkeras_special_cases.py)

#### Training example:
- [training_example.py](training_example.py)

#### All possible nets for `Classifiers.get()` method: 
`'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101',
        'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
        'inceptionresnetv2', 'inceptionv3',  'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
        'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetv2-b0',
        'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-s', 'efficientnetv2-m',
        'efficientnetv2-l', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'`

### Convert imagenet weights (2D -> 3D)

Code to convert 2D imagenet weights to 3D variant is available here: [convert_imagenet_weights_to_3D_models.py](convert_imagenet_weights_to_3D_models.py). 

### How to choose input shape

If initial 2D model had shape (512, 512, 3) then you can use shape (D, H, W, 3) where `D * H * W ~= 512*512`, so something like
(64, 64, 64, 3) will be ok.

Training with single NVIDIA 1080Ti (11 GB) worked with:
* DenseNet121, DenseNet169 and ResNet50 with shape (96, 128, 128, 3) and batch size 6
* DenseNet201 with shape (96, 128, 128, 3) and batch size 5
* ResNet18 with shape (128, 160, 160, 3) and batch size 6

### Additional features

#### Pooling
Default pooling/stride size for 3D models is set equal to 2. You can change it for your needs using parameter 
 `stride_size`. Example:
 
 ```python
from classification_models_3D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(
    input_shape=(224, 224, 224, 3),
    stride_size=4,
    kernel_size=3, 
    weights=None
)
```

`stride_size` can be: 
- single integer. Example: `4`
- tuple of size 5 (if you didn't change `repetition` parameter). Example: `(2, 2, 4, 2, 2)`
- tuple of tuples. Example: `(
(2, 2, 1), (2, 2, 4), (2, 2, 2), (2, 1, 2), (2, 4, 2),  
)`. Each number in `(2, 2, 1)` control stride of individual dimension.

#### More blocks

* For some models like (resnet, resnext, senet, vgg16, vgg19, densenet) it's possible to change number of blocks/poolings. 
For example if you want to make more poolings overall. You can do it like that:

 ```python
from classification_models_3D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(
    input_shape=(128, 128, 128, 3),
    include_top=False,
    weights=None,
    stride_size=(1, 1, 2, 2, 2, 2, 2, 2),
    repetitions=(2, 2, 2, 2, 2, 2, 2),
    init_filters=16,
)
```

- **Note 1**: Since number of filters grows 2 times, you can set initial number of filters with `init_filters` parameter.
- **Note 2**: There is no `imagenet` weights for models which were modified this way. 

### Related repositories

 * [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) - original 2D repo
 * [volumentations](https://github.com/ZFTurbo/volumentations) - 3D augmentations
 * [segmentation models 3D](https://github.com/ZFTurbo/segmentation_models_3D) - models for segmentation in 3D
 * [driven_data_repo](https://github.com/ZFTurbo/DrivenData-Alzheimer-Research-1st-place-solution) - code for training and inference on real dataset
 
### Unresolved problems

* There is no DepthwiseConv3D layer in keras, so repo used custom layer from [this repo](https://github.com/alexandrosstergiou/keras-DepthwiseConv3D) by [@alexandrosstergiou]( https://github.com/alexandrosstergiou/keras-DepthwiseConv3D) which can be slower than native implementation. 
* There is no imagenet weights for 'inceptionresnetv2' and 'inceptionv3'.
 
### Description
 
This code was used to get 1st place in [DrivenData: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/leaderboard/) competition.
 
More details on ArXiv: https://arxiv.org/abs/2104.01687

## Citation

For more details, please refer to the publication: https://doi.org/10.1016/j.compbiomed.2021.105089

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```
