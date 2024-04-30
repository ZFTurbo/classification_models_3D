# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""EfficientNet models for Keras.

Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)
"""

from .. import get_submodules_from_kwargs
from ..weights import load_model_weights
from keras.src.utils import file_utils

import os
import copy
import math

from keras.applications import imagenet_utils
from keras import models
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from ..models._DepthwiseConv3D import DepthwiseConv3D
from keras.src.legacy.backend import int_shape


DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

layers = VersionAwareLayers()

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)

  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For EfficientNet, input preprocessing is included as part of the model
  (as a `Rescaling` layer), and thus
  `tf.keras.applications.efficientnet.preprocess_input` is actually a
  pass-through function. EfficientNet models expect their inputs to be float
  tensors of pixels with values in the [0-255] range.

  Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded. Defaults to 'imagenet'.
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
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to 'softmax'.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""

def correct_pad_3d(inputs, kernel_size):
    """Returns a tuple for zero-padding for 3D convolution with downsampling.

      Args:
        inputs: Input tensor.
        kernel_size: An integer or tuple/list of 2 integers.

      Returns:
        A tuple.
      """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = int_shape(inputs)[img_dim:(img_dim + 3)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2, 1 - input_size[2] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
        (correct[2] - adjust[2], correct[2]),
    )


def EfficientNet(
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation='swish',
        blocks_args='default',
        model_name='efficientnet',
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

      Args:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

      Returns:
        A `keras.Model` instance.

      Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
      """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS

    if not (weights in {'imagenet', None} or file_utils.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

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

    bn_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.Rescaling(1. / 255.)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding3D(
        padding=correct_pad_3d(x, 3),
        name='stem_conv_pad')(x)
    x = layers.Conv3D(
        round_filters(32),
        3,
        strides=stride_size[0],
        padding='valid',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))

    strides_count = 1
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            elif args['strides'] > 1:
                args['strides'] = stride_size[strides_count]
                strides_count += 1
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1

    # Build top
    x = layers.Conv3D(
        round_filters(1280),
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name='top_conv'
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name='predictions'
        )(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D(name='max_pool')(x)

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


def block(
        inputs,
        activation='swish',
        drop_rate=0.,
        name='',
        filters_in=32,
        filters_out=16,
        kernel_size=3,
        strides=1,
        expand_ratio=1,
        se_ratio=0.,
        id_skip=True,
):
    """An inverted residual block.

      Args:
          inputs: input tensor.
          activation: activation function.
          drop_rate: float between 0 and 1, fraction of the input units to drop.
          name: string, block label.
          filters_in: integer, the number of input filters.
          filters_out: integer, the number of output filters.
          kernel_size: integer, the dimension of the convolution window.
          strides: integer, the stride of the convolution.
          expand_ratio: integer, scaling coefficient for the input filters.
          se_ratio: float between 0 and 1, fraction to squeeze the input filters.
          id_skip: boolean.

      Returns:
          output tensor for the block.
      """
    bn_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv3D(
            filters,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'expand_conv')(
            inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding3D(
            padding=correct_pad_3d(x, kernel_size),
            name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = DepthwiseConv3D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'dwconv'
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling3D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se_shape = (filters, 1, 1, 1)
        else:
            se_shape = (1, 1, 1, filters)
        se = layers.Reshape(se_shape, name=name + 'se_reshape')(se)
        se = layers.Conv3D(
            filters_se,
            1,
            padding='same',
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_reduce'
        )(se)
        se = layers.Conv3D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv3D(
        filters_out,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'project_conv'
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x


def EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.0,
        1.0,
        224,
        0.2,
        model_name='efficientnetb0',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.0,
        1.1,
        240,
        0.2,
        model_name='efficientnetb1',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.1,
        1.2,
        260,
        0.3,
        model_name='efficientnetb2',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.2,
        1.4,
        300,
        0.3,
        model_name='efficientnetb3',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.4,
        1.8,
        380,
        0.4,
        model_name='efficientnetb4',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB5(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.6,
        2.2,
        456,
        0.4,
        model_name='efficientnetb5',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB6(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        1.8,
        2.6,
        528,
        0.5,
        model_name='efficientnetb6',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


def EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        stride_size=2,
        classifier_activation='softmax',
        **kwargs
):
    return EfficientNet(
        2.0,
        3.1,
        600,
        0.5,
        model_name='efficientnetb7',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        stride_size=stride_size,
        classifier_activation=classifier_activation,
        **kwargs
    )


EfficientNetB0.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB0')
EfficientNetB1.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB1')
EfficientNetB2.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB2')
EfficientNetB3.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB3')
EfficientNetB4.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB4')
EfficientNetB5.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB5')
EfficientNetB6.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB6')
EfficientNetB7.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB7')


def preprocess_input(x, data_format=None, **kwargs):  # pylint: disable=unused-argument
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the efficientnet model
    implementation. Users are no longer required to call this method to normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it,
      it defaults to "channels_last").{mode}

    Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
