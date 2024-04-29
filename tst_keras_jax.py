# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import sys
    import os

    # For this test, make sure that only the tested framework is available
    sys.modules['tensorflow'] = None
    sys.modules['torch'] = None

    gpu_use = 0
    print(f"GPU use: {gpu_use}")
    os.environ["KERAS_BACKEND"] = "jax"
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
    include_top = False
    use_weights = None
    list_of_models = Classifiers.models_names()
    list_of_excepted_models = [ "efficientnetb0", "efficientnetb1", "efficientnetb2",
                                "efficientnetb3", "efficientnetb4", "efficientnetb5",
                                "efficientnetb6", "efficientnetb7", "efficientnetv2-b0",
                                "efficientnetv2-b1", "efficientnetv2-b2", "efficientnetv2-b3",
                                "efficientnetv2-s", "efficientnetv2-m", "efficientnetv2-l", ]
    for type in list_of_models:
        if type not in list_of_excepted_models:
            modelPoint, _ = Classifiers.get(type)
            model = modelPoint(input_shape=(128, 128, 128, 3), include_top=include_top, weights=use_weights)
            print(model.summary())
            print(get_model_memory_usage(1, model), 'GB')
            K.clear_session()


if __name__ == '__main__':
    tst_keras()
