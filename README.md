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

### Installation 

`pip install classification-models-3D`

### Examples 

##### Loading model with `imagenet` weights:

```python
# for keras
from classification_models_3D.keras import Classifiers

# for tensorflow.keras
# from classification_models_3D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(input_shape=(128, 128, 128, 3), weights='imagenet')
```

All possible nets for `Classifiers.get()` method: `'resnet18, 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2'`

### Convert imagenet weights (2D -> 3D)

Code to convert 2D imagenet weights to 3D variant is available here: [convert_imagenet_weights_to_3D_models.py](convert_imagenet_weights_to_3D_models.py). Weights were obtained with TF2, but works OK with Keras + TF1 as well.

### How to choose input shape

If initial 2D model had shape (512, 512, 3) then you can use shape (D, H, W, 3) where `D * H * W ~= 512*512`, so something like
(64, 64, 64, 3) will be ok.

Training with single NVIDIA 1080Ti (11 GB) worked with:
* DenseNet121, DenseNet169 and ResNet50 with shape (96, 128, 128, 3) and batch size 6
* DenseNet201 with shape (96, 128, 128, 3) and batch size 5
* ResNet18 with shape (128, 160, 160, 3) and batch size 6

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

If you find this code useful, please cite it as:
```
@InProceedings{RSolovyev_2021_stalled,
  author = {Solovyev, Roman and Kalinin, Alexandr A. and Gabruseva, Tatiana},
  title = {3D Convolutional Neural Networks for Stalled Brain Capillary Detection},
  booktitle = {Arxiv: 2104.01687},
  month = {April},
  year = {2021}
}
```
