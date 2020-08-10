# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 1
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def tst_keras():
    # for keras
    from keras import __version__
    from keras import backend as K
    from classification_models_3D.keras import Classifiers

    print('Keras version: {}'.format(__version__))
    include_top = False
    use_weights = 'imagenet'
    list_of_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50',
                 'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
                 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
                 'inceptionresnetv2', 'inceptionv3']
    for type in list_of_models:
        modelPoint, preprocess_input = Classifiers.get(type)
        model = modelPoint(input_shape=(128, 128, 128, 3), include_top=include_top, weights=use_weights)
        print(model.summary())
        K.clear_session()


if __name__ == '__main__':
    tst_keras()
