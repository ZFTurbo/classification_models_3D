# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import sys
    import os

    # For this test, make sure that only the tested framework is available
    sys.modules['tensorflow'] = None
    sys.modules['jax'] = None

    gpu_use = 0
    print(f"GPU use: {gpu_use}")
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_use}"


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K
    from keras.src.utils import summary_utils

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = summary_utils.count_params(model.trainable_weights)
    non_trainable_count = summary_utils.count_params(model.non_trainable_weights)

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def tst_keras():
    # for keras
    from keras import __version__
    from keras import backend as K
    from classification_models_3D.kkeras import Classifiers

    print(f"Keras version {__version__} using {K.backend()} backend")
    if 1:
        type = 'densenet121'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
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
        K.clear_session()

    if 1:
        type = 'inceptionresnetv2'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
        model = modelPoint(
            input_shape=(299, 299, 299, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 2, 2, 4, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        K.clear_session()

    if 1:
        type = 'inceptionv3'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
        model = modelPoint(
            input_shape=(299, 299, 299, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        K.clear_session()

    if 1:
        type = 'mobilenet'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
        model = modelPoint(
            input_shape=(224, 224, 224, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        K.clear_session()

    if 1:
        type = 'mobilenetv2'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
        model = modelPoint(
            input_shape=(224, 224, 224, 3),
            include_top=False,
            weights=None,
            stride_size=(2, 4, 2, 2, 2),
        )
        print(model.summary())
        print(get_model_memory_usage(1, model), 'GB')
        K.clear_session()

    if 1:
        type = 'resnet18'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
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
        K.clear_session()

    if 1:
        type = 'resnext50'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
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
        K.clear_session()

    if 1:
        type = 'seresnet101'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
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
        K.clear_session()

    if 1:
        type = 'vgg16'
        print(f"Go for {type}")
        modelPoint, _ = Classifiers.get(type)
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
        K.clear_session()


if __name__ == '__main__':
    tst_keras()
