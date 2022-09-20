# Release notes
All notable changes to this project will be documented in this file.

##  v1.0.7

Added convNeXt 3D models

##  v1.0.6

Padding fixes for `mobilenetv2`, `inceptionresnetv2` and `inceptionv3`. It's needed for correct work with segmentation models.

##  v1.0.4

- Added EfficientNet and EfficientNet v2 models (with converted imagenet weights)
- Fixed error in converter which skip bias for imagenet weights
- `include_top` parameter now False by default. It was made because imageNet weights now only available for `include_top = False`. 
 If use `include_top=True` and `classes=<N>`, you must use `weights=None`.
- New converted imagenet weights available
- Added parameter `stride_size` to control how strides/poolings are made. By default it's equal to `2`. Now it's possible to set individual stride for each stage of model. 
For example:
```python
stride_size = [
    (1, 2, 1),
    (2, 2, 4),
    (2, 2, 4),
    (2, 4, 2),
    (2, 2, 2),
]
```
Here each tuple control individual stride/pooling. While each tuple control stride for each dimension.
Strides doesn't affect model structure and you can use `imagenet` weights with such modified models.

- For some models (resnet, resnext, senet, densenet, vgg16, vgg19) it's possible to increase number of blocks using `repetition` parameter. It can be useful if you need to add more poolings and layers. `imagenet` weights won't work for modified models.
- Minimum TF version bumped to 2.8.0
