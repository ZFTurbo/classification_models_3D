# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

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


def tst_keras():
    # for tensorflow.keras
    from tensorflow import __version__
    from tensorflow.compat.v1 import reset_default_graph
    from classification_models_3D.tfkeras import Classifiers

    print('Tensorflow version: {}'.format(__version__))
    include_top = False
    use_weights = 'imagenet'
    # use_weights = None
    list_of_models = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50',
        'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101', 'senet154', 'resnext50', 'resnext101',
        'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
        'inceptionresnetv2', 'inceptionv3',  'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
        'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetv2-b0',
        'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnetv2-s', 'efficientnetv2-m',
        'efficientnetv2-l', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'
    ]
    for type in list_of_models:
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(128, 128, 128, 3),
            include_top=include_top,
            stride_size=2,
            weights=use_weights
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()


if __name__ == '__main__':
    tst_keras()
