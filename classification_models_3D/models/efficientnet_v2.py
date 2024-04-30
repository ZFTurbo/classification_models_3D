# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
"""EfficientNet V2 models for Keras.

Reference:
- [EfficientNetV2: Smaller Models and Faster Training](
    https://arxiv.org/abs/2104.00298) (ICML 2021)
"""

import os
import copy
import math

from .. import get_submodules_from_kwargs
from ..weights import load_model_weights
from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras import models
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
import tensorflow.compat.v2 as tf
from keras.src.utils import file_utils
from ..models._DepthwiseConv3D import DepthwiseConv3D


DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [{
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
    }, {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.0,
        "strides": 2,
        "conv_type": 1,
    }, {
        "conv_type": 1,
        "expand_ratio": 4,
        "input_filters": 48,
        "kernel_size": 3,
        "num_repeat": 4,
        "output_filters": 64,
        "se_ratio": 0,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "input_filters": 64,
        "kernel_size": 3,
        "num_repeat": 6,
        "output_filters": 128,
        "se_ratio": 0.25,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 128,
        "kernel_size": 3,
        "num_repeat": 9,
        "output_filters": 160,
        "se_ratio": 0.25,
        "strides": 1,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 160,
        "kernel_size": 3,
        "num_repeat": 15,
        "output_filters": 256,
        "se_ratio": 0.25,
        "strides": 2,
    }],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal"
    }
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [EfficientNetV2: Smaller Models and Faster Training](
      https://arxiv.org/abs/2104.00298) (ICML 2021)

  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For EfficientNetV2, by default input preprocessing is included as a part of the
  model (as a `Rescaling` layer), and thus
  `tf.keras.applications.efficientnet_v2.preprocess_input` is actually a
  pass-through function. In this use case, EfficientNetV2 models expect their inputs
  to be float tensors of pixels with values in the [0-255] range.
  At the same time, preprocessing as a part of the model (i.e. `Rescaling`
  layer) can be disabled by setting `include_preprocessing` argument to False.
  With preprocessing disabled EfficientNetV2 models expect their inputs to be float
  tensors of pixels with values in the [-1, 1] range.

  Args:
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
      `"imagenet"` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False.
      It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`. Defaults to None.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `"avg"` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 2D tensor.
      - `"max"` means that global max pooling will
          be applied.
    classes: Optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Defaults to 1000 (number of
      ImageNet classes).
    classifier_activation: A string or callable. The activation function to use
      on the `"top"` layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      Defaults to `"softmax"`.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def MBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """MBConv block: Mobile Inverted Residual Bottleneck."""
    bn_axis = -1 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        # Expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv3D(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=bn_momentum,
                name=name + "expand_bn",
            )(x)
            x = layers.Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise conv
        x = DepthwiseConv3D(
            kernel_size=kernel_size,
            strides=strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=name + "dwconv2",
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "bn")(x)
        x = layers.Activation(activation, name=name + "activation")(x)

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1, 1)
            else:
                se_shape = (1, 1, 1, filters)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

            se = layers.Conv3D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = layers.Conv3D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = layers.multiply([x, se], name=name + "se_excite")

            # Output phase
            x = layers.Conv3D(
                filters=output_filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "project_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)

            if strides == 1 and input_filters == output_filters:
                if survival_probability:
                    x = layers.Dropout(
                        survival_probability,
                        noise_shape=(None, 1, 1, 1, 1),
                        name=name + "drop",
                    )(x)
                x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
    bn_axis = -1 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv3D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                data_format="channels_last",
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=bn_axis, momentum=bn_momentum, name=name + "expand_bn")(x)
            x = layers.Activation(
                activation=activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1, 1)
            else:
                se_shape = (1, 1, 1, filters)

            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

            se = layers.Conv3D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = layers.Conv3D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase:
        x = layers.Conv3D(
            output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)
        if expand_ratio == 1:
            x = layers.Activation(
                activation=activation, name=name + "project_activation")(x)

        # Residual:
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                x = layers.Dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1, 1),
                    name=name + "drop",
                )(x)
            x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def EfficientNetV2(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args="default",
    model_name="efficientnetv2",
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    """Instantiates the EfficientNetV2 architecture using given scaling coefficients.

      Args:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        min_depth: integer, minimum number of filters.
        bn_momentum: float. Momentum parameter for Batch Normalization layers.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected layer at the top of the
          network.
        weights: one of `None` (random initialization), `"imagenet"` (pre-training
          on ImageNet), or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) or
          numpy array to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if `include_top` is
          False. It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction when `include_top` is
          `False`. - `None` means that the output of the model will be the 4D tensor
          output of the last convolutional layer. - "avg" means that global average
          pooling will be applied to the output of the last convolutional layer, and
          thus the output of the model will be a 2D tensor. - `"max"` means that
          global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
          specified if `include_top` is True, and if no `weights` argument is
          specified.
        classifier_activation: A string or callable. The activation function to use
          on the `"top"` layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the `"top"` layer.
        include_preprocessing: Boolean, whether to include the preprocessing layer
          (`Rescaling`) at the bottom of the network. Defaults to `True`.

      Returns:
        A `keras.Model` instance.

      Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `"softmax"` or `None` when
          using a pretrained top layer.
      """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError("The `weights` argument should be either "
                         "`None` (random initialization), `imagenet` "
                         "(pre-training on ImageNet), "
                         "or the path to the weights file to be loaded."
                         f"Received: weights={weights}")

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError("If using `weights` as `'imagenet'` with `include_top`"
                         " as true, `classes` should be 1000"
                         f"Received: classes={classes}")

    # if stride_size is scalar make it tuple of length 5 with elements tuple of size 3
    # (stride for each dimension for more flexibility)
    if type(stride_size) not in (tuple, list):
        stride_size = [
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
        ]
    else:
        stride_size = list(stride_size)

    if len(stride_size) != 5:
        print('Error: stride_size length must be exactly 5')
        return None

    for i in range(len(stride_size)):
        if type(stride_size[i]) not in (tuple, list):
            stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = -1 if backend.image_data_format() == "channels_last" else 1

    x = img_input

    if include_preprocessing:
        # Apply original V1 preprocessing for Bx variants
        # if number of channels allows it
        num_channels = input_shape[-1]
        if model_name.split("-")[-1].startswith("b") and num_channels == 3:
            x = layers.Rescaling(scale=1. / 255)(x)
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
                axis=bn_axis,
            )(x)
        else:
            x = layers.Rescaling(scale=1. / 128.0, offset=-1)(x)

    # Build stem
    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )
    x = layers.Conv3D(
        filters=stem_filters,
        kernel_size=3,
        strides=stride_size[0],
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=bn_momentum,
        name="stem_bn",
    )(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    strides_count = 1
    for (i, args) in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)

        # Determine which conv type to use:
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient)
        for j in range(repeats):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]
            elif args["strides"] > 1:
                args["strides"] = stride_size[strides_count]
                strides_count += 1

            x = block(
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args,
            )(x)

    # Build top
    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor
    )
    x = layers.Conv3D(
        filters=top_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name="top_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=bn_momentum,
        name="top_bn",
    )(x)
    x = layers.Activation(activation=activation, name="top_activation")(x)

    if include_top:
        x = layers.GlobalAveragePooling3D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(0),
            name="predictions")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling3D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling3D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_name, weights, classes, include_top, **kwargs)

    return model


def EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        model_name="efficientnetv2-b0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2B1(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        model_name="efficientnetv2-b1",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2B2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        model_name="efficientnetv2-b2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2B3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=300,
        model_name="efficientnetv2-b3",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=384,
        model_name="efficientnetv2-s",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2M(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-m",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


def EfficientNetV2L(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    stride_size=2,
    classifier_activation="softmax",
    include_preprocessing=True,
    **kwargs,
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-l",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        **kwargs,
    )


EfficientNetV2B0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B0")
EfficientNetV2B1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B1")
EfficientNetV2B2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B2")
EfficientNetV2B3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B3")
EfficientNetV2S.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2S")
EfficientNetV2M.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2M")
EfficientNetV2L.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2L")


def preprocess_input(x, data_format=None, **kwargs):  # pylint: disable=unused-argument
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the EfficientNetV2 model
    implementation. Users are no longer required to call this method to normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it, it
      defaults to "channels_last").{mode}

    Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
