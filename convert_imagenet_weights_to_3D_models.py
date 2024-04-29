# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

try:
    # tf keras
    from tensorflow.keras import backend as K
    from classification_models.tfkeras import Classifiers as Classifiers_2D
    from classification_models_3D.tfkeras import Classifiers as Classifiers_3D
    print('Use TF keras...')
except:
    # keras
    from keras import backend as K
    from classification_models.keras import Classifiers as Classifiers_2D
    from classification_models_3D.kkeras import Classifiers as Classifiers_3D
    print('Use keras...')

import os
import glob
import hashlib

from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import EfficientNetB1
from keras.applications.efficientnet import EfficientNetB2
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.efficientnet import EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet_v2 import *
from keras.applications.convnext import ConvNeXtTiny
from keras.applications.convnext import ConvNeXtSmall
from keras.applications.convnext import ConvNeXtBase
from keras.applications.convnext import ConvNeXtLarge
from keras.applications.convnext import ConvNeXtXLarge


MODELS_PATH = './'
OUTPUT_PATH_CONVERTER = MODELS_PATH + 'converter/'
if not os.path.isdir(OUTPUT_PATH_CONVERTER):
    os.mkdir(OUTPUT_PATH_CONVERTER)


def get_model_memory_usage(batch_size, model):
    import numpy as np

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def convert_weights(m2, m3, out_path, target_channel):
    print('Start: {}'.format(m2.name))
    for i in range(len(m2.layers)):
        layer_2D = m2.layers[i]
        layer_3D = m3.layers[i]
        print('Extract for [{}]: {} {}'.format(i, layer_2D.__class__.__name__, layer_2D.name))
        print('Set for [{}]: {} {}'.format(i, layer_3D.__class__.__name__, layer_3D.name))

        if layer_2D.name != layer_3D.name:
            print('Warning: different names!')

        weights_2D = layer_2D.get_weights()
        weights_3D = layer_3D.get_weights()
        if layer_2D.__class__.__name__ == 'Conv2D' or \
                layer_2D.__class__.__name__ == 'DepthwiseConv2D':
            print(type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_3D[0].shape)
            print(layer_2D.output_shape)
            print(layer_3D.output_shape)
            weights_3D[0][...] = 0
            if target_channel == 2:
                for j in range(weights_3D[0].shape[2]):
                    weights_3D[0][:, :, j, :, :] = weights_2D[0] / weights_3D[0].shape[2]
            if target_channel == 1:
                for j in range(weights_3D[0].shape[1]):
                    weights_3D[0][:, j, :, :, :] = weights_2D[0] / weights_3D[0].shape[1]
            else:
                for j in range(weights_3D[0].shape[0]):
                    weights_3D[0][j, :, :, :, :] = weights_2D[0] / weights_3D[0].shape[0]

            # Bias
            if len(weights_3D) > 1:
                print(weights_3D[1].shape, weights_2D[1].shape)
                weights_3D[1] = weights_2D[1][:weights_3D[1].shape[0]]

            m3.layers[i].set_weights(weights_3D)
        elif layer_2D.__class__.__name__ == 'Sequential' and 'convnext' in layer_2D.name:
            print('Convnext', type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_3D[0].shape)
            print(layer_2D.output_shape)
            print(layer_3D.output_shape)

            if 'downsampling' in layer_2D.name:
                index_w = 2
                index_b = 3
                layer_norm_0 = 0
                layer_norm_1 = 1
            else:
                index_w = 0
                index_b = 1
                layer_norm_0 = 2
                layer_norm_1 = 3

            weights_3D[index_w][...] = 0
            if target_channel == 2:
                for j in range(weights_3D[index_w].shape[2]):
                    weights_3D[index_w][:, :, j, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[2]
            if target_channel == 1:
                for j in range(weights_3D[index_w].shape[1]):
                    weights_3D[index_w][:, j, :, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[1]
            else:
                for j in range(weights_3D[index_w].shape[0]):
                    weights_3D[index_w][j, :, :, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[0]

            # Bias
            if len(weights_3D) > 1:
                print(weights_3D[index_b].shape, weights_2D[index_b].shape)
                weights_3D[index_b] = weights_2D[index_b][:weights_3D[index_b].shape[0]]

            # layer norm
            weights_3D[layer_norm_0] = weights_2D[layer_norm_0]
            weights_3D[layer_norm_1] = weights_2D[layer_norm_1]

            m3.layers[i].set_weights(weights_3D)
        elif layer_2D.__class__.__name__ == 'Normalization' and i == 2:
            if len(weights_3D) == 0:
                # Effnet v2 (it's in parameters)
                pass
        else:
            m3.layers[i].set_weights(weights_2D)

    m3.save(out_path)


def convert_models():
    include_top = False
    target_channel = 0
    shape_size_3D = (64, 64, 64, 3)
    # shape_size_3D = (32, 7*32, 7*32, 3)
    shape_size_2D = (224, 224, 3)
    list_to_check = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101',
        'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
        'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
        'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetv2-b0',
        'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-s', 'efficientnetv2-m',
        'efficientnetv2-l',  'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'
    ]
    list_to_check = [
       'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'
    ]

    for t in list_to_check:
        out_path = MODELS_PATH + 'converter/{}_inp_channel_{}_tch_{}_top_{}.h5'.format(t, shape_size_3D[-1], target_channel, include_top)
        if os.path.isfile(out_path):
            print('Already exists: {}!'.format(out_path))
            continue

        model3D, preprocess_input = Classifiers_3D.get(t)
        model3D = model3D(include_top=include_top,
                              weights=None,
                              input_shape=shape_size_3D,
                              pooling='avg', )
        mem = get_model_memory_usage(1, model3D)
        print('Model 3D: {} Mem single: {:.2f}'.format(t, mem))

        if t in ['efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
                 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']:
            func = {
                'efficientnetb0': EfficientNetB0,
                'efficientnetb1': EfficientNetB1,
                'efficientnetb2': EfficientNetB2,
                'efficientnetb3': EfficientNetB3,
                'efficientnetb4': EfficientNetB4,
                'efficientnetb5': EfficientNetB5,
                'efficientnetb6': EfficientNetB6,
                'efficientnetb7': EfficientNetB7,
            }

            model2D = func[t](
                include_top=include_top,
                weights='imagenet',
                input_shape=shape_size_2D,
                pooling='avg',
            )
        elif t in ['efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3',
                   'efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l']:

            func = {
                'efficientnetv2-b0': EfficientNetV2B0,
                'efficientnetv2-b1': EfficientNetV2B1,
                'efficientnetv2-b2': EfficientNetV2B2,
                'efficientnetv2-b3': EfficientNetV2B3,
                'efficientnetv2-s': EfficientNetV2S,
                'efficientnetv2-m': EfficientNetV2M,
                'efficientnetv2-l': EfficientNetV2L,
            }

            model2D = func[t](
                include_top=include_top,
                weights='imagenet',
                input_shape=shape_size_2D,
                pooling='avg',
            )
        elif t in ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge']:

            func = {
                'convnext_tiny': ConvNeXtTiny,
                'convnext_small': ConvNeXtSmall,
                'convnext_base': ConvNeXtBase,
                'convnext_large': ConvNeXtLarge,
                'convnext_xlarge': ConvNeXtXLarge,
            }

            model2D = func[t](
                include_top=include_top,
                weights='imagenet',
                input_shape=shape_size_2D,
                pooling='avg',
            )
        else:
            model2D, preprocess_input = Classifiers_2D.get(t)
            model2D = model2D(
                include_top=include_top,
                weights='imagenet',
                input_shape=shape_size_2D,
                pooling='avg',
            )
        mem = get_model_memory_usage(1, model2D)
        print('Model 2D: {} Mem single: {:.2f}'.format(t, mem))

        convert_weights(model2D, model3D, out_path, target_channel=target_channel)
        K.clear_session()


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def gen_text_with_links():
    list_to_check = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101',
        'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
        'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
        'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetv2-b0',
        'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-s', 'efficientnetv2-m',
        'efficientnetv2-l', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'
    ]
    for model_name in list_to_check:
        files = glob.glob('./converter/{}_*.h5'.format(model_name))
        for f in files:
            file_name = os.path.basename(f)
            arr = file_name[:-3].split('_')
            m5 = md5(f)

            print('# {}'.format(model_name))
            print('{')
            print('    \'model\': \'{}\','.format(model_name))
            print('    \'dataset\': \'imagenet\','.format(model_name))
            print('    \'classes\': 1000,'.format(model_name))
            print('    \'include_top\': {},'.format(arr[-1]))
            print('    \'url\': \'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/{}\','.format(file_name))
            print('    \'name\': \'{}\','.format(file_name))
            print('    \'md5\': \'{}\','.format(m5))
            print('},')


if __name__ == '__main__':
    convert_models()
    gen_text_with_links()
