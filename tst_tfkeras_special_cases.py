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
    if 0:
        type = 'densenet121'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(128, 128, 128, 2),
            include_top=False,
            weights=None,
            stride_size=(1, 1, 2, 2, 2, 2, 2),
            kernel_size=3,
            repetitions=(6, 12, 24, 16, 8, 4),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'efficientnetb0'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(128, 128, 128, 2),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'efficientnetv2-b0'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(128, 128, 128, 2),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'inceptionresnetv2'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(299, 299, 299, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'inceptionv3'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(299, 299, 299, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'mobilenet'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(224, 224, 224, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'mobilenetv2'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(224, 224, 224, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'resnet18'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(256, 256, 256, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2, 2, 2, 2),
            repetitions=(2, 2, 2, 2, 2, 2, 2),
            init_filters=16,
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'resnext50'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(256, 256, 256, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2, 2, 2),
            repetitions=(2, 2, 2, 2, 2, 2),
            init_filters=64,
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 0:
        type = 'seresnet101'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(224, 224, 224, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 4, 2, 2, 2),
            repetitions=(2, 2, 2, 2, 2),
            init_filters=32,
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()

    if 1:
        type = 'vgg16'
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(
            input_shape=(256, 256, 256, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 4, 2, 2),
            repetitions=(2, 2, 3, 2, 2),
            init_filters=64,
            max_filters=1024,
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        reset_default_graph()


if __name__ == '__main__':
    tst_keras()
